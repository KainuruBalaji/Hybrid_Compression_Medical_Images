"""Module 1: Image Preprocessing — load, grayscale, pad, block division, merge."""

from typing import Tuple

import cv2
import numpy as np


def create_example_image(output_path: str = "example_input.png", size: int = 256) -> np.ndarray:
    """Create a simple MRI-like grayscale example image for the demo."""
    image = np.full((size, size), 18, dtype=np.uint8)
    center = (size // 2, size // 2)

    cv2.ellipse(image, center, (88, 112), 0, 0, 360, 65, -1)
    cv2.ellipse(image, center, (74, 96), 0, 0, 360, 105, -1)
    cv2.ellipse(image, center, (48, 64), 0, 0, 360, 150, -1)
    cv2.ellipse(image, (center[0] - 24, center[1] - 8), (18, 26), 15, 0, 360, 190, -1)
    cv2.ellipse(image, (center[0] + 28, center[1] + 6), (22, 30), -10, 0, 360, 210, -1)
    cv2.circle(image, (center[0] - 6, center[1] + 40), 12, 232, -1)

    gradient = np.tile(np.linspace(0, 28, size, dtype=np.uint8), (size, 1))
    image = cv2.add(image, gradient)
    image = cv2.GaussianBlur(image, (7, 7), 0)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 6, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, noisy_image)
    return noisy_image


def load_grayscale_image(image_path: str) -> np.ndarray:
    """Read an image and convert it to grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def pad_image_to_block_size(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image so dimensions are exact multiples of block_size."""
    height, width = image.shape
    padded_height = int(np.ceil(height / block_size) * block_size)
    padded_width = int(np.ceil(width / block_size) * block_size)

    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[:height, :width] = image
    return padded, (height, width)


def split_into_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """Split a padded image into non-overlapping blocks of size block_size×block_size."""
    blocks = []
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            blocks.append(image[row:row + block_size, col:col + block_size])
    return np.asarray(blocks, dtype=np.uint8)


def merge_blocks(blocks: np.ndarray, image_shape: Tuple[int, int], block_size: int) -> np.ndarray:
    """Merge reconstructed blocks back into a single image."""
    height, width = image_shape
    reconstructed = np.zeros((height, width), dtype=np.float32)

    block_index = 0
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            reconstructed[row:row + block_size, col:col + block_size] = blocks[block_index]
            block_index += 1

    return np.clip(reconstructed, 0, 255).astype(np.uint8)
