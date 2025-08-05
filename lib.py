import numpy as np
from PIL import Image as im
import cv2
from sklearn.neighbors import NearestNeighbors
import io
import base64

def stream_mnist_assignment(digits, image_bytes, scale=1.0, k=10):
    arr = np.array(im.open(image_bytes).convert("L"))
    h, w = arr.shape
    new_h = int(h * scale)
    new_w = int(w * scale)
    arr_resized = cv2.resize(arr, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
    h, w = arr_resized.shape
    mnist_h = h // 28
    mnist_w = w // 28
    yield mnist_w, mnist_h

    arr_resized = arr_resized[: mnist_h * 28, : mnist_w * 28]
    blocks = []
    for my in range(mnist_h):
        for mx in range(mnist_w):
            block = arr_resized[
                my * 28 : (my + 1) * 28, mx * 28 : (mx + 1) * 28
            ]
            blocks.append(block)
    blocks = np.array(blocks)
    num_blocks = blocks.shape[0]
    block_indices = np.arange(num_blocks)
    np.random.shuffle(block_indices)
    digits_flat = digits.reshape(digits.shape[0], -1).astype(np.float32)
    blocks_flat = blocks.reshape(num_blocks, -1).astype(np.float32)

    _, indices = (
        NearestNeighbors(n_neighbors=k, algorithm="auto")
        .fit(digits_flat)
        .kneighbors(blocks_flat)
    )

    used = np.zeros(digits.shape[0], dtype=bool)
    matching_digits_idx = np.zeros(num_blocks, dtype=int)



    mosaic = np.zeros((mnist_h * 28, mnist_w * 28), dtype=np.uint8)

    for block_idx in block_indices:
        for kth in range(k):
            idx = indices[block_idx, kth]
            if not used[idx]:
                matching_digits_idx[block_idx] = idx
                used[idx] = True
                break
        else:
            available = np.where(~used)[0]
            if available.size > 0:
                matching_digits_idx[block_idx] = available[0]
                used[available[0]] = True
            else:
                matching_digits_idx[block_idx] = 0

        block_y = block_idx // mnist_w
        block_x = block_idx % mnist_w
        mnist_block = digits[matching_digits_idx[block_idx]]

        mosaic[
            block_y * 28 : (block_y + 1) * 28,
            block_x * 28 : (block_x + 1) * 28
        ] = mnist_block

        buf = io.BytesIO()
        im.fromarray(mnist_block).save(buf, format="PNG")
        block_hex = buf.getvalue().hex()
        yield block_x, block_y, block_hex

    buf = io.BytesIO()
    im.fromarray(mosaic).save(buf, format="PNG")
    final_img_bytes = buf.getvalue()
    final_img_b64 = base64.b64encode(final_img_bytes).decode('ascii')
    yield "final", final_img_b64