from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import io
import base64
from lib import stream_mnist_assignment

(train_X, _), (test_X, _) = mnist.load_data()
digits = np.concatenate([train_X, test_X], axis=0)

app = FastAPI()

@app.get("/")
def index():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/process")
async def ws_process(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the image as bytes
        data = await websocket.receive_bytes()
        gen = stream_mnist_assignment(digits, io.BytesIO(data), scale=1.0, k=10)
        meta = next(gen)
        mnist_w, mnist_h = meta
        await websocket.send_json({"type": "meta", "w": mnist_w, "h": mnist_h})

        for item in gen:
            if item[0] == "final":
                await websocket.send_json({"type": "final", "b64": item[1]})
            else:
                block_x, block_y, block_hex = item
                await websocket.send_json({
                    "type": "block",
                    "x": int(block_x),
                    "y": int(block_y),
                    "hex": block_hex
                })
    except WebSocketDisconnect:
        pass