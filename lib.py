import numpy as np
from PIL import Image as im
import cv2
from sklearn.neighbors import NearestNeighbors
import io
import base64

def resize_image(image_bytes, scale=1.0):
    arr = np.array(im.open(image_bytes).convert("L"))
    h, w = arr.shape
    new_h = int(h * scale)
    new_w = int(w * scale)

    arr_resized = cv2.resize(arr, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC) 
    mnist_h = new_h // 28
    mnist_w = new_w // 28
    return mnist_w, mnist_h, arr_resized[: mnist_h * 28, : mnist_w * 28]

def stream_mnist_assignment(digits, mnist_w, mnist_h, arr_resized, k=10):
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
    # np.random.shuffle(block_indices)
    digits_flat = digits.reshape(digits.shape[0], -1).astype(np.float32)
    blocks_flat = blocks.reshape(num_blocks, -1).astype(np.float32)

    _, indices = (
        NearestNeighbors(n_neighbors=k, algorithm="auto")
        .fit(digits_flat)
        .kneighbors(blocks_flat)
    )

    used = np.zeros(digits.shape[0], dtype=bool)

    for block_idx in block_indices:
        block_mnist_idx = None
        for kth in range(k):
            idx = indices[block_idx, kth]
            if not used[idx]:
                block_mnist_idx = idx
                used[idx] = True
                break
        else:
            # TODO: this else is completely wrong
            available = np.where(~used)[0]
            if available.size > 0:
                block_mnist_idx = available[0]
                used[available[0]] = True
                print("not zero", len(available))
            else:
                print('zero')
                block_mnist_idx = 0

        block_y = block_idx // mnist_w
        block_x = block_idx % mnist_w
        mnist_block = digits[block_mnist_idx]

        yield block_x, block_y, mnist_block