import os
import requests
import urllib3
import gradio as gr

# Disable warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Environment variables
URL = os.getenv("INFERENCE_ENDPOINT")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")


# Inference call
def predict(
    distance_from_last_transaction,
    ratio_to_median_purchase_price,
    used_chip,
    used_pin_number,
    online_order,
):
    payload = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": [1, 5],
                "datatype": "FP32",
                "data": [
                    [
                        distance_from_last_transaction,
                        ratio_to_median_purchase_price,
                        used_chip,
                        used_pin_number,
                        online_order,
                    ]
                ],
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(URL, json=payload, headers=headers, verify=False)
    prediction = response.json()["outputs"][0]["data"][0]
    return "Fraud" if prediction >= 0.995 else "Not fraud"


# Gradio app
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Distance from Last Transaction"),
        gr.Number(label="Ratio to Median Purchase Price"),
        gr.Number(label="Used Chip"),
        gr.Number(label="Used PIN Number"),
        gr.Number(label="Online Order"),
    ],
    outputs="textbox",
    examples=[[0.311, 1.946, 1.0, 100.0, 0.0], [175.989, 0.856, 0.0, 0.0, 1.0]],
    title="Predict Credit Card Fraud",
)

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
