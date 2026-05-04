"""DCTAE: Hybrid Image Compression Using DCT and Autoencoder.

This package implements the DCTAE paper's pipeline:
    1. Block division and entropy-based ROI/Non-ROI classification
    2. DCT compression for high-entropy ROI blocks
    3. Sparse autoencoder compression for low-entropy Non-ROI blocks
    4. Reconstruction and evaluation
"""

from .preprocessing import (
    create_example_image,
    load_grayscale_image,
    pad_image_to_block_size,
    split_into_blocks,
    merge_blocks,
)
from .entropy import compute_block_entropy, entropy_calculation
from .thresholding import otsu_threshold
from .dct_compression import jpeg_quantization_matrix, dct_compress
from .autoencoder import (
    build_dense_autoencoder,
    build_conv_autoencoder,
    train_autoencoder,
    compress_blocks,
    decompress_blocks,
)
from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_compression_ratio,
    compute_space_saving,
    compute_bpp,
    compute_all_metrics,
)
from .visualization import visualize_results, plot_entropy_histogram, visualize_simple
from .reconstruction import reconstruction_pipeline

__all__ = [
    "create_example_image",
    "load_grayscale_image",
    "pad_image_to_block_size",
    "split_into_blocks",
    "merge_blocks",
    "compute_block_entropy",
    "entropy_calculation",
    "otsu_threshold",
    "jpeg_quantization_matrix",
    "dct_compress",
    "build_dense_autoencoder",
    "build_conv_autoencoder",
    "train_autoencoder",
    "compress_blocks",
    "decompress_blocks",
    "compute_psnr",
    "compute_ssim",
    "compute_compression_ratio",
    "compute_space_saving",
    "compute_bpp",
    "compute_all_metrics",
    "visualize_results",
    "plot_entropy_histogram",
    "visualize_simple",
    "reconstruction_pipeline",
]
