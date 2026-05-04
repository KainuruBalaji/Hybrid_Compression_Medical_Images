import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


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
    """Read an image and convert it to grayscale, matching the paper's first step."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def pad_image_to_block_size(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad image so that it can be partitioned into non-overlapping blocks."""
    height, width = image.shape
    padded_height = int(np.ceil(height / block_size) * block_size)
    padded_width = int(np.ceil(width / block_size) * block_size)

    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[:height, :width] = image
    return padded, (height, width)


def split_into_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """Split a padded image into non-overlapping blocks."""
    blocks = []
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            blocks.append(image[row:row + block_size, col:col + block_size])
    return np.asarray(blocks, dtype=np.uint8)


def merge_blocks(blocks: np.ndarray, image_shape: Tuple[int, int], block_size: int) -> np.ndarray:
    """Merge blocks back into a single image."""
    height, width = image_shape
    reconstructed = np.zeros((height, width), dtype=np.float32)

    block_index = 0
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            reconstructed[row:row + block_size, col:col + block_size] = blocks[block_index]
            block_index += 1

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def entropy_calculation(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and normalized entropy values for each block."""
    entropies = []
    for block in blocks:
        histogram = np.bincount(block.flatten(), minlength=256).astype(np.float64)
        probabilities = histogram / np.sum(histogram)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        entropies.append(entropy)

    entropies = np.asarray(entropies, dtype=np.float32)
    total_entropy = float(np.sum(entropies))
    normalized = entropies / total_entropy if total_entropy > 0 else np.zeros_like(entropies)
    return entropies, normalized


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute an Otsu threshold over block entropy values."""
    if values.size == 0:
        return 0.0

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        return min_value

    histogram, bin_edges = np.histogram(values, bins=bins, range=(min_value, max_value))
    histogram = histogram.astype(np.float64)
    probabilities = histogram / np.sum(histogram)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cumulative_probabilities = np.cumsum(probabilities)
    cumulative_means = np.cumsum(probabilities * bin_centers)
    global_mean = cumulative_means[-1]

    numerator = (global_mean * cumulative_probabilities - cumulative_means) ** 2
    denominator = cumulative_probabilities * (1.0 - cumulative_probabilities)
    denominator[denominator == 0] = np.nan
    between_class_variance = numerator / denominator

    best_index = int(np.nanargmax(between_class_variance))
    return float(bin_centers[best_index])


def jpeg_quantization_matrix(block_size: int) -> np.ndarray:
    """Create a JPEG-style quantization matrix sized to the current block."""
    base_matrix = np.array(
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

    tile_factor = int(np.ceil(block_size / 8))
    tiled_matrix = np.tile(base_matrix, (tile_factor, tile_factor))
    return tiled_matrix[:block_size, :block_size]


def dct_compress(block: np.ndarray, quant_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compress and reconstruct an ROI block with DCT, quantization, and IDCT."""
    centered_block = block.astype(np.float32) - 128.0
    dct_coefficients = cv2.dct(centered_block)
    quantized_coefficients = np.round(dct_coefficients / quant_matrix)
    dequantized_coefficients = quantized_coefficients * quant_matrix
    reconstructed = cv2.idct(dequantized_coefficients) + 128.0
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed.astype(np.float32), quantized_coefficients.astype(np.int16)


def autoencoder_model(
    input_dim: int,
    latent_dim: int = 32,
    sparsity_weight: float = 1e-4,
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build a small sparse autoencoder for non-ROI blocks."""
    inputs = tf.keras.Input(shape=(input_dim,), name="input_block")
    hidden = tf.keras.layers.Dense(128, activation="relu", name="encoder_hidden")(inputs)
    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="sigmoid",
        activity_regularizer=tf.keras.regularizers.L1(sparsity_weight),
        name="latent_code",
    )(hidden)
    decoder_hidden = tf.keras.layers.Dense(128, activation="relu", name="decoder_hidden")(latent)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(decoder_hidden)

    autoencoder = tf.keras.Model(inputs, outputs, name="hybrid_autoencoder")
    encoder = tf.keras.Model(inputs, latent, name="encoder")

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="decoder_input")
    hidden_layer = autoencoder.get_layer("decoder_hidden")(latent_inputs)
    decoder_outputs = autoencoder.get_layer("reconstruction")(hidden_layer)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder, decoder


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Use skimage when available, otherwise fall back to the standard PSNR formula."""
    if SKIMAGE_AVAILABLE:
        return float(peak_signal_noise_ratio(original, reconstructed, data_range=255))

    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Use skimage when available, otherwise compute a grayscale SSIM approximation."""
    if SKIMAGE_AVAILABLE:
        return float(structural_similarity(original, reconstructed, data_range=255))

    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(original, kernel, sigma)
    mu2 = cv2.GaussianBlur(reconstructed, kernel, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(original * original, kernel, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(reconstructed * reconstructed, kernel, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * reconstructed, kernel, sigma) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def visualize_results(
    original: np.ndarray,
    reconstructed: np.ndarray,
    roi_map: np.ndarray,
    output_path: str,
) -> None:
    """Save original image, ROI map, and reconstructed image for quick inspection."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(roi_map, cmap="gray")
    axes[1].set_title("ROI / Non-ROI Map")
    axes[2].imshow(reconstructed, cmap="gray")
    axes[2].set_title("Reconstructed")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def reconstruction_pipeline(
    image_path: str,
    block_size: int = 16,
    latent_dim: int = 32,
    epochs: int = 40,
    batch_size: int = 16,
    output_dir: str = "outputs",
) -> Dict[str, object]:
    """Run the full DCTAE pipeline described in the paper."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original_image = load_grayscale_image(image_path)
    padded_image, original_shape = pad_image_to_block_size(original_image, block_size)
    blocks = split_into_blocks(padded_image, block_size)

    entropies, normalized_entropies = entropy_calculation(blocks)
    threshold = otsu_threshold(entropies)
    roi_flags = entropies > threshold

    quant_matrix = jpeg_quantization_matrix(block_size)

    non_roi_blocks = blocks[~roi_flags]
    autoencoder = encoder = decoder = None
    reconstructed_non_roi_blocks = np.empty((0, block_size, block_size), dtype=np.float32)
    quantized_latent = np.empty((0, latent_dim), dtype=np.uint8)

    if len(non_roi_blocks) > 0:
        non_roi_vectors = non_roi_blocks.astype(np.float32).reshape(len(non_roi_blocks), -1) / 255.0
        autoencoder, encoder, decoder = autoencoder_model(
            input_dim=block_size * block_size,
            latent_dim=latent_dim,
            sparsity_weight=1e-4,
        )
        autoencoder.fit(
            non_roi_vectors,
            non_roi_vectors,
            epochs=epochs,
            batch_size=min(batch_size, max(1, len(non_roi_vectors))),
            verbose=0,
            shuffle=True,
        )

        latent_codes = encoder.predict(non_roi_vectors, verbose=0)
        quantized_latent = np.round(latent_codes * 255.0).astype(np.uint8)
        dequantized_latent = quantized_latent.astype(np.float32) / 255.0
        reconstructed_vectors = decoder.predict(dequantized_latent, verbose=0)
        reconstructed_non_roi_blocks = reconstructed_vectors.reshape(-1, block_size, block_size) * 255.0

    reconstructed_blocks = np.zeros((len(blocks), block_size, block_size), dtype=np.float32)
    dct_nonzero_coefficients = 0
    non_roi_index = 0

    for block_index, block in enumerate(blocks):
        if roi_flags[block_index]:
            reconstructed_block, quantized_coefficients = dct_compress(block, quant_matrix)
            reconstructed_blocks[block_index] = reconstructed_block
            dct_nonzero_coefficients += int(np.count_nonzero(quantized_coefficients))
        else:
            reconstructed_blocks[block_index] = reconstructed_non_roi_blocks[non_roi_index]
            non_roi_index += 1

    padded_reconstructed = merge_blocks(reconstructed_blocks, padded_image.shape, block_size)
    reconstructed_image = padded_reconstructed[:original_shape[0], :original_shape[1]]

    pixel_roi_map = np.zeros(padded_image.shape, dtype=np.uint8)
    block_grid_width = padded_image.shape[1] // block_size
    for block_index, is_roi in enumerate(roi_flags):
        row = (block_index // block_grid_width) * block_size
        col = (block_index % block_grid_width) * block_size
        pixel_roi_map[row:row + block_size, col:col + block_size] = 255 if is_roi else 0
    roi_map = pixel_roi_map[:original_shape[0], :original_shape[1]]

    original_bits = original_image.size * 8
    roi_bits = max(1, dct_nonzero_coefficients) * 16
    non_roi_bits = max(1, quantized_latent.size) * 8 if len(non_roi_blocks) > 0 else 0
    compressed_bits = roi_bits + non_roi_bits
    compression_ratio = float(original_bits / max(compressed_bits, 1))

    psnr_value = compute_psnr(original_image, reconstructed_image)
    ssim_value = compute_ssim(original_image, reconstructed_image)

    reconstructed_image_path = output_path / "reconstructed_image.png"
    roi_map_path = output_path / "roi_map.png"
    comparison_path = output_path / "comparison.png"

    cv2.imwrite(str(reconstructed_image_path), reconstructed_image)
    cv2.imwrite(str(roi_map_path), roi_map)
    visualize_results(original_image, reconstructed_image, roi_map, str(comparison_path))

    return {
        "original_image": original_image,
        "reconstructed_image": reconstructed_image,
        "roi_map": roi_map,
        "entropies": entropies,
        "normalized_entropies": normalized_entropies,
        "threshold": threshold,
        "roi_block_count": int(np.sum(roi_flags)),
        "non_roi_block_count": int(np.sum(~roi_flags)),
        "psnr": psnr_value,
        "ssim": ssim_value,
        "compression_ratio": compression_ratio,
        "skimage_available": SKIMAGE_AVAILABLE,
    }


def main() -> None:
    """Run the pipeline on a user image or fall back to the generated example image."""
    parser = argparse.ArgumentParser(description="Hybrid image compression using DCT and Autoencoder")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the input image. If omitted, an example image is generated and used.",
    )
    parser.add_argument("--block-size", type=int, default=16, help="Non-overlapping block size")
    parser.add_argument("--latent-dim", type=int, default=32, help="Autoencoder latent dimension")
    parser.add_argument("--epochs", type=int, default=40, help="Autoencoder training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Autoencoder training batch size")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for saved outputs")
    args = parser.parse_args()

    if args.image_path is None:
        example_image_path = "example_input.png"
        if not Path(example_image_path).exists():
            create_example_image(example_image_path)
        input_image_path = example_image_path
    else:
        input_image_path = args.image_path
        if not Path(input_image_path).exists():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

    results = reconstruction_pipeline(
        image_path=input_image_path,
        block_size=args.block_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print("Hybrid Image Compression using DCT and Autoencoder")
    print(f"Input image          : {input_image_path}")
    print(f"Entropy threshold    : {results['threshold']:.4f}")
    print(f"ROI blocks           : {results['roi_block_count']}")
    print(f"Non-ROI blocks       : {results['non_roi_block_count']}")
    print(f"PSNR                 : {results['psnr']:.4f} dB")
    print(f"SSIM                 : {results['ssim']:.4f}")
    print(f"Compression Ratio    : {results['compression_ratio']:.4f}")
    print(f"skimage available    : {results['skimage_available']}")
    print(
        "Saved outputs        : "
        f"{args.output_dir}/reconstructed_image.png, "
        f"{args.output_dir}/roi_map.png, "
        f"{args.output_dir}/comparison.png"
    )


if __name__ == "__main__":
    main()
