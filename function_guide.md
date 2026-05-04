# Project Functions Guide

This document lists every code file in the project and explains the purpose of each function.

## `dctae/preprocessing.py`
- **`create_example_image`**: Generates a synthetic grayscale image for testing.
- **`load_grayscale_image`**: Reads an image file from disk and converts it to grayscale.
- **`pad_image_to_block_size`**: Pads image dimensions with zeros to be exact multiples of the block size.
- **`split_into_blocks`**: Slices the padded image into non-overlapping blocks.
- **`merge_blocks`**: Reassembles processed blocks back into a complete image.

## `dctae/entropy.py`
- **`compute_block_entropy`**: Calculates the Shannon entropy for a single image block.
- **`entropy_calculation`**: Computes raw and normalized entropies for a list of blocks.

## `dctae/thresholding.py`
- **`otsu_threshold`**: Calculates Otsu's optimal threshold over block entropy values to separate ROI from Non-ROI.

## `dctae/dct_compression.py`
- **`jpeg_quantization_matrix`**: Creates a scaled JPEG-style quantization matrix for the specific block size.
- **`dct_compress`**: Compresses and reconstructs an ROI block using Discrete Cosine Transform (DCT), quantization, and Inverse DCT (IDCT).

## `dctae/autoencoder.py`
- **`build_dense_autoencoder`**: Builds an improved deep dense sparse autoencoder model for compressing smooth blocks.
- **`build_conv_autoencoder`**: Builds a convolutional autoencoder model for compressing smooth blocks.
- **`train_autoencoder`**: Trains the autoencoder on Non-ROI blocks using early stopping to prevent overfitting.
- **`compress_blocks`**: Encodes blocks into a low-dimensional latent representation and quantizes them to 8-bit integers.
- **`decompress_blocks`**: Dequantizes latent codes and decodes them back to reconstructed image blocks.
- **`prepare_training_data`**: Reshapes and normalizes block data before feeding it to the autoencoder for training.

## `dctae/metrics.py`
- **`compute_psnr`**: Computes the Peak Signal-to-Noise Ratio (PSNR) between the original and reconstructed images.
- **`compute_ssim`**: Computes the Structural Similarity Index Measure (SSIM) between the images.
- **`compute_compression_ratio`**: Calculates the compression ratio (original size divided by compressed size).
- **`compute_space_saving`**: Calculates the space saving percentage based on the compression ratio.
- **`compute_bpp`**: Calculates the Bits Per Pixel (BPP) metric for the compressed image.
- **`compute_all_metrics`**: Aggregates all evaluation metrics into a single dictionary.

## `dctae/visualization.py`
- **`plot_entropy_histogram`**: Plots the distribution of block entropies and visualizes the Otsu threshold split.
- **`visualize_results`**: Generates a 4-panel figure comparing the original image, ROI map, reconstructed image, and entropy histogram.
- **`visualize_simple`**: Generates a simple 3-panel comparison figure.

## `dctae/reconstruction.py`
- **`reconstruction_pipeline`**: Orchestrates the complete hybrid image compression pipeline, connecting all stages together.

## `main.py`
- **`main`**: CLI entry point that handles arguments and runs the pipeline for a single image.

## `batch_evaluate.py`
- **`batch_evaluate`**: Processes an entire folder of images, computes metrics for each, and outputs a summary CSV.
- **`main`**: CLI entry point that handles arguments for batch processing.
