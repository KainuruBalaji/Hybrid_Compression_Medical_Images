"""Module 6: Evaluation Metrics — PSNR, SSIM, CR, SS, BPP."""

from typing import Dict

import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio.

    PSNR = 20 · log₁₀(MAX / √MSE)

    Higher PSNR → better reconstruction quality.
    Typical values: 25–40 dB for lossy compression.
    """
    if SKIMAGE_AVAILABLE:
        return float(peak_signal_noise_ratio(original, reconstructed, data_range=255))

    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural Similarity Index Measure.

    SSIM = (2μ₁μ₂ + C₁)(2σ₁₂ + C₂) / ((μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂))

    Range: [-1, 1]. Higher → more perceptually similar.
    """
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
    """Compression Ratio: original_bits / compressed_bits.

    Compressed bits include:
        - ROI: non-zero quantized DCT coefficients × 16 bits each
        - Non-ROI: quantized latent codes × 8 bits each
        - Overhead: 1 bit per block for ROI/Non-ROI flag
    """
    original_bits = original_size * 8
    roi_bits = max(1, dct_nonzero_count) * 16
    non_roi_bits = max(1, latent_size) * 8
    flag_bits = num_blocks
    compressed_bits = roi_bits + non_roi_bits + flag_bits
    return float(original_bits / max(compressed_bits, 1))


def compute_space_saving(compression_ratio: float) -> float:
    """Space Saving percentage.

    SS = (1 - 1/CR) × 100

    E.g., CR=4.0 → SS=75% (75% of storage saved)
    """
    if compression_ratio <= 0:
        return 0.0
    return (1.0 - 1.0 / compression_ratio) * 100.0


def compute_bpp(
    original_size: int,
    dct_nonzero_count: int,
    latent_size: int,
    num_blocks: int,
) -> float:
    """Bits Per Pixel — average number of bits used per pixel after compression.

    BPP = compressed_bits / total_pixels

    Lower BPP → better compression.
    Uncompressed grayscale = 8.0 BPP.
    Typical lossy compression: 0.5–2.0 BPP.
    """
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
    """Compute all evaluation metrics at once."""
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
