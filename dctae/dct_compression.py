"""Module 4: DCT Compression — 2D DCT, JPEG quantization, and IDCT for ROI blocks."""

from typing import Tuple

import cv2
import numpy as np


_BASE_QUANT_MATRIX = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


def jpeg_quantization_matrix(block_size: int) -> np.ndarray:
    """Create a JPEG-style quantization matrix scaled to the given block size.

    The standard 8×8 JPEG quantization table is tiled to cover the requested
    block size, then cropped.
    """
    tile_factor = int(np.ceil(block_size / 8))
    tiled = np.tile(_BASE_QUANT_MATRIX, (tile_factor, tile_factor))
    return tiled[:block_size, :block_size]


def dct_compress(block: np.ndarray, quant_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compress and reconstruct an ROI block using DCT + quantization + IDCT.

    Steps:
        1. Level-shift: subtract 128 to center values around 0
        2. Forward 2D DCT
        3. Quantize: round(DCT / Q_matrix)
        4. Dequantize: quantized × Q_matrix
        5. Inverse 2D DCT + add 128
        6. Clip to [0, 255]

    Returns:
        reconstructed: the IDCT-reconstructed block (float32)
        quantized_coefficients: the quantized DCT coefficients (int16)
    """
    centered = block.astype(np.float32) - 128.0
    dct_coeffs = cv2.dct(centered)
    quantized = np.round(dct_coeffs / quant_matrix)
    dequantized = quantized * quant_matrix
    reconstructed = cv2.idct(dequantized) + 128.0
    return np.clip(reconstructed, 0, 255).astype(np.float32), quantized.astype(np.int16)
