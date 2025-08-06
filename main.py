import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import HTMLResponse

from tensorflow.keras.datasets import mnist

from lib import resize_image
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
        data = await websocket.receive_bytes()
        mnist_w, mnist_h, arr_resized = resize_image(io.BytesIO(data), scale=1.0)
        await websocket.send_json({"type": "meta", "w": mnist_w, "h": mnist_h})


        mosaic = np.zeros((mnist_h * 28, mnist_w * 28), dtype=np.uint8)
        for block_x, block_y, mnist_block in stream_mnist_assignment(digits, mnist_w, mnist_h, arr_resized, k=1000):
            mosaic[
                block_y * 28 : (block_y + 1) * 28,
                block_x * 28 : (block_x + 1) * 28
            ] = mnist_block
            
            block_buf = io.BytesIO()
            Image.fromarray(mnist_block).save(block_buf, format="PNG")
            # progress indicator, blocks are shown as they are processed
            await websocket.send_json({
                "type": "block",
                "b64": base64.b64encode(block_buf.getvalue()).decode('ascii'),
                "x": int(block_x),
                "y": int(block_y)
            })
        
        # download the actual image
        final_buf = io.BytesIO()
        Image.fromarray(mosaic).save(final_buf, format="PNG")
        final_img_bytes = final_buf.getvalue()
        await websocket.send_json({
            "type": "final", 
            "b64": base64.b64encode(final_img_bytes).decode('ascii')
        })
    except WebSocketDisconnect:
        pass