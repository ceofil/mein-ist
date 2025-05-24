import numpy as np
from PIL import Image as im
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm


def concat_2d(blocks, h, w):
    lines = []
    for y in range(h):
        images_in_line = []
        for x in range(w):
            img = blocks[w * y + x]
            images_in_line.append(img)
        line = np.hstack(images_in_line)
        lines.append(line)
    return np.vstack(lines)

def save_img_array(arr, filename):
    print(f'saved {filename} with shape {arr.shape}')
    im.fromarray(arr).save(filename)


def test(_digits, input_file, output_file, new_h):
    digits = _digits.copy()
    image = cv2.imread(input_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = new_h / h
    new_w = int(w * scale)
    gray = cv2.resize(gray, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
    plt.imshow(gray)
    h, w = gray.shape
    mnist_h = h // 28
    mnist_w = w // 28
    h = mnist_h * 28
    w = mnist_w * 28
    print(w, h, scale)

    matching_digits = []

    mnist_coords = []
    for my in range(mnist_h):
        for mx in range(mnist_w):
            mnist_coords.append((mx, my))

    excluded = np.zeros(len(digits))
    for mx, my in tqdm(mnist_coords):
        block = gray[my * 28 : (my+1) * 28, mx * 28: (mx + 1) * 28]
        matching_idx = min(
            range(len(digits)), 
            key=lambda idx: np.linalg.norm(block - digits[idx]) + excluded[idx]
        )
        matching_digits.append(digits[matching_idx])
        excluded[matching_idx] = 2**24
        # digits = np.delete(digits, matching_idx, axis=0)

    big_image = concat_2d(matching_digits, mnist_h, mnist_w)
    save_img_array(big_image, output_file)