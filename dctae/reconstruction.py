"""Module 8: Reconstruction Pipeline — full DCTAE pipeline combining all modules."""

import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

from .preprocessing import load_grayscale_image, pad_image_to_block_size, split_into_blocks, merge_blocks
from .entropy import entropy_calculation
from .thresholding import otsu_threshold
from .dct_compression import jpeg_quantization_matrix, dct_compress
from .autoencoder import (
    build_dense_autoencoder,
    build_conv_autoencoder,
    train_autoencoder,
    compress_blocks,
    decompress_blocks,
    prepare_training_data,
)
from .metrics import compute_all_metrics
from .visualization import visualize_results, plot_entropy_histogram


def reconstruction_pipeline(
    image_path: str,
    block_size: int = 16,
    latent_dim: int = 32,
    epochs: int = 100,
    batch_size: int = 16,
    model_type: str = "dense",
    sparsity_weight: float = 1e-4,
    output_dir: str = "outputs",
) -> Dict[str, object]:
    """Run the full DCTAE hybrid compression pipeline.

    Pipeline stages:
        1. Load grayscale image and pad to block-aligned dimensions
        2. Split into non-overlapping blocks
        3. Compute Shannon entropy for each block
        4. Find Otsu threshold on entropy values
        5. Classify blocks: entropy > threshold → ROI, else → Non-ROI
        6. Compress ROI blocks with DCT + JPEG quantization
        7. Compress Non-ROI blocks with sparse autoencoder
        8. Merge reconstructed blocks into final image
        9. Evaluate metrics (PSNR, SSIM, CR, SS)
        10. Save visualizations

    Args:
        image_path: path to input image
        block_size: size of non-overlapping blocks (default 16)
        latent_dim: autoencoder bottleneck dimension (default 32)
        epochs: max training epochs for autoencoder (default 100)
        batch_size: training batch size (default 16)
        model_type: "dense" or "conv" autoencoder architecture
        sparsity_weight: L1 regularization weight on latent layer
        output_dir: directory for saved outputs
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: Load and pad
    original_image = load_grayscale_image(image_path)
    padded_image, original_shape = pad_image_to_block_size(original_image, block_size)

    # Stage 2: Block division
    blocks = split_into_blocks(padded_image, block_size)

    # Stage 3: Entropy calculation
    entropies, normalized_entropies = entropy_calculation(blocks)

    # Stage 4: Otsu thresholding
    threshold = otsu_threshold(entropies)

    # Stage 5: ROI / Non-ROI classification
    roi_flags = entropies > threshold

    # Stage 6: DCT compression for ROI blocks
    quant_matrix = jpeg_quantization_matrix(block_size)

    # Stage 7: Autoencoder compression for Non-ROI blocks
    non_roi_blocks = blocks[~roi_flags]
    quantized_latent = np.empty((0, latent_dim), dtype=np.uint8)

    if len(non_roi_blocks) > 0:
        train_data = prepare_training_data(non_roi_blocks, block_size, model_type)

        if model_type == "conv":
            autoencoder, encoder, decoder = build_conv_autoencoder(
                block_size=block_size,
                latent_dim=latent_dim,
                sparsity_weight=sparsity_weight,
            )
        else:
            autoencoder, encoder, decoder = build_dense_autoencoder(
                input_dim=block_size * block_size,
                latent_dim=latent_dim,
                sparsity_weight=sparsity_weight,
            )

        train_autoencoder(autoencoder, train_data, epochs=epochs, batch_size=batch_size)

        quantized_latent, _ = compress_blocks(encoder, non_roi_blocks, block_size, model_type)
        reconstructed_non_roi = decompress_blocks(decoder, quantized_latent, block_size, model_type)
    else:
        reconstructed_non_roi = np.empty((0, block_size, block_size), dtype=np.float32)

    # Stage 8: Merge all blocks
    reconstructed_blocks = np.zeros((len(blocks), block_size, block_size), dtype=np.float32)
    dct_nonzero_count = 0
    non_roi_idx = 0

    for i, block in enumerate(blocks):
        if roi_flags[i]:
            recon_block, quant_coeffs = dct_compress(block, quant_matrix)
            reconstructed_blocks[i] = recon_block
            dct_nonzero_count += int(np.count_nonzero(quant_coeffs))
        else:
            reconstructed_blocks[i] = reconstructed_non_roi[non_roi_idx]
            non_roi_idx += 1

    padded_reconstructed = merge_blocks(reconstructed_blocks, padded_image.shape, block_size)
    reconstructed_image = padded_reconstructed[:original_shape[0], :original_shape[1]]

    # Build ROI map for visualization
    pixel_roi_map = np.zeros(padded_image.shape, dtype=np.uint8)
    grid_width = padded_image.shape[1] // block_size
    for i, is_roi in enumerate(roi_flags):
        row = (i // grid_width) * block_size
        col = (i % grid_width) * block_size
        pixel_roi_map[row:row + block_size, col:col + block_size] = 255 if is_roi else 0
    roi_map = pixel_roi_map[:original_shape[0], :original_shape[1]]

    # Stage 9: Metrics
    metrics = compute_all_metrics(
        original_image,
        reconstructed_image,
        dct_nonzero_count,
        quantized_latent.size,
        len(blocks),
    )

    # Stage 10: Save outputs
    cv2.imwrite(str(output_path / "reconstructed_image.png"), reconstructed_image)
    cv2.imwrite(str(output_path / "roi_map.png"), roi_map)

    visualize_results(
        original_image, reconstructed_image, roi_map,
        entropies, threshold,
        str(output_path / "comparison.png"),
    )
    plot_entropy_histogram(
        entropies, threshold,
        str(output_path / "entropy_histogram.png"),
    )

    return {
        "original_image": original_image,
        "reconstructed_image": reconstructed_image,
        "roi_map": roi_map,
        "entropies": entropies,
        "normalized_entropies": normalized_entropies,
        "threshold": threshold,
        "roi_block_count": int(np.sum(roi_flags)),
        "non_roi_block_count": int(np.sum(~roi_flags)),
        "total_blocks": len(blocks),
        "model_type": model_type,
        **metrics,
    }
