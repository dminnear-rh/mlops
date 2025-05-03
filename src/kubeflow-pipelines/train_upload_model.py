import os

from kfp import compiler, dsl, kubernetes
from kfp.dsl import InputPath, OutputPath


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2024a-20240523"
)
def get_data(
    train_data_output_path: OutputPath(), validate_data_output_path: OutputPath()
):
    import urllib.request

    print("starting download...")
    print("downloading training data")
    url = "https://raw.githubusercontent.com/rh-aiservices-bu/fraud-detection/main/data/train.csv"
    urllib.request.urlretrieve(url, train_data_output_path)
    print("train data downloaded")
    print("downloading validation data")
    url = "https://raw.githubusercontent.com/rh-aiservices-bu/fraud-detection/main/data/validate.csv"
    urllib.request.urlretrieve(url, validate_data_output_path)
    print("validation data downloaded")


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2024a-20240523",
    packages_to_install=["onnx==1.17.0", "onnxruntime==1.19.2", "tf2onnx==1.16.1"],
)
def train_model(
    train_data_input_path: InputPath(),
    validate_data_input_path: InputPath(),
    model_output_path: OutputPath(),
):
    # This version builds a large network (4096 units per layer) in float64 to
    # force high memory usage during training.

    import pickle
    from pathlib import Path

    import numpy as np
    import onnx
    import pandas as pd
    import tensorflow as tf
    import tf2onnx
    from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
    from keras.models import Sequential
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight

    # Use float64 to double tensor size
    tf.keras.backend.set_floatx("float64")

    # Columns: 0..6 are features, 7 is the label
    feature_indexes = list(range(7))
    label_indexes = [7]

    # Load data
    X_train_df = pd.read_csv(train_data_input_path)
    y_train_df = X_train_df.iloc[:, label_indexes]
    X_train_df = X_train_df.iloc[:, feature_indexes]

    X_val_df = pd.read_csv(validate_data_input_path)
    y_val_df = X_val_df.iloc[:, label_indexes]
    X_val_df = X_val_df.iloc[:, feature_indexes]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values).astype("float64")
    X_val = scaler.transform(X_val_df.values).astype("float64")

    Path("artifact").mkdir(parents=True, exist_ok=True)
    pickle.dump(scaler, open("artifact/scaler.pkl", "wb"))

    # Class weights for imbalance
    cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train_df), y=y_train_df.values.ravel()
    )
    class_weights = {i: w for i, w in enumerate(cw)}

    # Build a large fully connected network
    UNITS = 4096
    model = Sequential(
        [
            Input(shape=(len(feature_indexes),), dtype="float64"),
            Dense(UNITS, activation="relu"),
            Dropout(0.3),
            Dense(UNITS),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(UNITS),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(UNITS),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(
        X_train,
        y_train_df,
        epochs=2,
        validation_data=(X_val, y_val_df),
        verbose=2,
        class_weight=class_weights,
    )

    # Export to ONNX
    spec = (
        tf.TensorSpec((None, len(feature_indexes)), tf.float64, name="dense_input"),
    )
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, model_output_path)


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2024a-20240523",
    packages_to_install=["boto3==1.35.55", "botocore==1.35.55"],
)
def upload_model(input_model_path: InputPath()):
    import os

    import boto3
    import botocore

    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    region_name = os.environ.get("AWS_DEFAULT_REGION")
    bucket_name = os.environ.get("AWS_S3_BUCKET")

    s3_key = os.environ.get("S3_KEY")

    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource(
        "s3",
        config=botocore.client.Config(signature_version="s3v4"),
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model_path, s3_key)


@dsl.pipeline(name=os.path.basename(__file__).replace(".py", ""))
def pipeline():
    get_data_task = get_data()
    train_data_csv_file = get_data_task.outputs["train_data_output_path"]
    validate_data_csv_file = get_data_task.outputs["validate_data_output_path"]

    train_model_task = train_model(
        train_data_input_path=train_data_csv_file,
        validate_data_input_path=validate_data_csv_file,
    )
    onnx_file = train_model_task.outputs["model_output_path"]

    upload_model_task = upload_model(input_model_path=onnx_file)

    upload_model_task.set_env_variable(name="S3_KEY", value="models/fraud/1/model.onnx")

    kubernetes.use_secret_as_env(
        task=upload_model_task,
        secret_name="aws-connection-my-storage",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            "AWS_S3_BUCKET": "AWS_S3_BUCKET",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
        },
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=__file__.replace(".py", ".yaml")
    )
