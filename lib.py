import numpy as np
from PIL import Image as im
from matplotlib import pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors
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

def test(digits, input_file, output_file, scale, k=10):
    image = cv2.imread(input_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_h = int(h * scale)
    new_w = int(w * scale)
    gray_resized = cv2.resize(gray, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
    h, w = gray_resized.shape
    mnist_h = h // 28
    mnist_w = w // 28
    h = mnist_h * 28
    w = mnist_w * 28
    gray_resized = gray_resized[:h, :w]

    blocks = []
    for my in range(mnist_h):
        for mx in range(mnist_w):
            block = gray_resized[my * 28 : (my+1) * 28, mx * 28: (mx + 1) * 28]
            blocks.append(block)
    blocks = np.array(blocks)
    num_blocks = blocks.shape[0]
    digits_flat = digits.reshape(digits.shape[0], -1).astype(np.float32)
    blocks_flat = blocks.reshape(num_blocks, -1).astype(np.float32)

    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(digits_flat)
    distances, indices = nn.kneighbors(blocks_flat)  # indices: (num_blocks, k)

    block_indices = np.arange(num_blocks)
    np.random.shuffle(block_indices)

    used = np.zeros(digits.shape[0], dtype=bool)
    matching_digits_idx = np.zeros(num_blocks, dtype=int)
    fallback_count = 0

    for i in tqdm(block_indices, desc="Assigning blocks (shuffled)"):
        for j in range(k):
            idx = indices[i, j]
            if not used[idx]:
                matching_digits_idx[i] = idx
                used[idx] = True
                break
        else:
            fallback_count += 1
            available = np.where(~used)[0]
            dists = np.sum((blocks_flat[i] - digits_flat[available]) ** 2, axis=1)
            idx_in_available = np.argmin(dists)
            idx = available[idx_in_available]
            matching_digits_idx[i] = idx
            used[idx] = True

    print(f"\nfallbacks used: {fallback_count} times")

    matching_digits = digits[matching_digits_idx]

    big_image = concat_2d(matching_digits, mnist_h, mnist_w)
    save_img_array(big_image, output_file)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Image (resized)")
    plt.imshow(gray_resized, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Output Image")
    plt.imshow(big_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
