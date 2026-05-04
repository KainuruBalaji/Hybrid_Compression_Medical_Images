

from typing import Dict

import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # Computes the Peak Signal-to-Noise Ratio (PSNR) between the original and reconstructed images.
    if SKIMAGE_AVAILABLE:
        return float(peak_signal_noise_ratio(original, reconstructed, data_range=255))

    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # Computes the Structural Similarity Index Measure (SSIM) between the images.
    if SKIMAGE_AVAILABLE:
        return float(structural_similarity(original, reconstructed, data_range=255))

    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(orig, kernel, sigma)
    mu2 = cv2.GaussianBlur(recon, kernel, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(orig * orig, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(recon * recon, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig * recon, kernel, sigma) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def compute_compression_ratio(
    original_size: int,
    dct_nonzero_count: int,
    latent_size: int,
    num_blocks: int,
) -> float:
    # Calculates the compression ratio (original size divided by compressed size).
    original_bits = original_size * 8
    roi_bits = max(1, dct_nonzero_count) * 16
    non_roi_bits = max(1, latent_size) * 8
    flag_bits = num_blocks
    compressed_bits = roi_bits + non_roi_bits + flag_bits
    return float(original_bits / max(compressed_bits, 1))


def compute_space_saving(compression_ratio: float) -> float:
    # Calculates the space saving percentage based on the compression ratio.
    if compression_ratio <= 0:
        return 0.0
    return (1.0 - 1.0 / compression_ratio) * 100.0


def compute_bpp(
    original_size: int,
    dct_nonzero_count: int,
    latent_size: int,
    num_blocks: int,
) -> float:
    # Calculates the Bits Per Pixel (BPP) metric for the compressed image.
    total_pixels = original_size  # grayscale: 1 byte per pixel
    roi_bits = max(1, dct_nonzero_count) * 16
    non_roi_bits = max(1, latent_size) * 8
    flag_bits = num_blocks
    compressed_bits = roi_bits + non_roi_bits + flag_bits
    return float(compressed_bits / max(total_pixels, 1))


def compute_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    dct_nonzero_count: int,
    latent_size: int,
    num_blocks: int,
) -> Dict[str, float]:
    # Aggregates all evaluation metrics into a single dictionary.
    psnr = compute_psnr(original, reconstructed)
    ssim = compute_ssim(original, reconstructed)
    cr = compute_compression_ratio(original.size, dct_nonzero_count, latent_size, num_blocks)
    ss = compute_space_saving(cr)
    bpp = compute_bpp(original.size, dct_nonzero_count, latent_size, num_blocks)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "compression_ratio": cr,
        "space_saving": ss,
        "bpp": bpp,
        "skimage_available": SKIMAGE_AVAILABLE,
    }
