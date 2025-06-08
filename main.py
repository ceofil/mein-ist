from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from tensorflow.keras.datasets import mnist
import numpy as np
import io
from lib import stream_mnist_assignment

(train_X, _), (test_X, _) = mnist.load_data()
digits = np.concatenate([train_X, test_X], axis=0)

app = FastAPI()

def stream_blocks_and_final(digits, image_bytes):
    gen = stream_mnist_assignment(digits, io.BytesIO(image_bytes), scale=.8, k=1000)
    meta = next(gen)
    mnist_w, mnist_h = meta
    yield f"event:meta\ndata:{mnist_w},{mnist_h}\n\n"

    for item in gen:
        if item[0] == "final":
            final_img_b64 = item[1]
            yield f"event:final\ndata:{final_img_b64}\n\n"
        else:
            block_x, block_y, block_hex = item
            yield f"event:block\ndata:{block_x},{block_y},{block_hex}\n\n"

@app.post("/api/process")
async def process_image(file: UploadFile):
    image_bytes = await file.read()
    return StreamingResponse(
        stream_blocks_and_final(digits, image_bytes),
        media_type="text/event-stream"
    )

@app.get("/")
def index():
    return HTMLResponse(open("index.html").read())