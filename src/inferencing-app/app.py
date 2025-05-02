import os

import gradio as gr
import requests
import urllib3

# Disable warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Environment variables
URL = os.getenv("INFERENCE_ENDPOINT")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")


# Inference call
def predict(
    distance_from_home,
    distance_from_last_transaction,
    ratio_to_median_purchase_price,
    repeat_retailer,
    used_chip,
    used_pin_number,
    online_order,
):
    payload = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": [1, 7],
                "datatype": "FP32",
                "data": [
                    [
                        distance_from_home,
                        distance_from_last_transaction,
                        ratio_to_median_purchase_price,
                        repeat_retailer,
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
        gr.Number(label="Distance from Home"),
        gr.Number(label="Distance from Last Transaction"),
        gr.Number(label="Ratio to Median Purchase Price"),
        gr.Radio([0, 1], label="Repeat Retailer"),
        gr.Radio([0, 1], label="Used Chip"),
        gr.Radio([0, 1], label="Used PIN Number"),
        gr.Radio([0, 1], label="Online Order"),
    ],
    outputs="textbox",
    examples=[
        [57.88, 0.31, 1.95, 1, 1, 0, 0],
        [15.69, 175.99, 0.86, 0, 0, 0, 1],
    ],
    title="Predict Credit Card Fraud",
    allow_flagging="never",
)

demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT)
