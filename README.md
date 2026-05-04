# DCTAE: Hybrid Image Compression Using DCT and Autoencoder

Implementation of the paper **"Hybrid Image Compression Using DCT and Autoencoder"** — a hybrid compression system that routes image blocks to different compressors based on their information content.

## How It Works

1. **Grayscale Conversion** — Input image is converted to grayscale
2. **Block Division** — Image is split into non-overlapping 16×16 blocks
3. **Entropy Calculation** — Shannon entropy computed per block
4. **Otsu Thresholding** — Automatic threshold separates ROI (high entropy) from Non-ROI (low entropy) blocks
5. **ROI → DCT Compression** — High-detail blocks compressed with 2D DCT + JPEG quantization
6. **Non-ROI → Autoencoder** — Smooth blocks compressed with a sparse autoencoder
7. **Reconstruction** — Merge all blocks into the final image
8. **Evaluation** — PSNR, SSIM, Compression Ratio, Space Saving

## Project Structure

```
MS_Project/
├── dctae/                          # Core package
│   ├── __init__.py                 # Package exports
│   ├── preprocessing.py            # Image loading, padding, block division
│   ├── entropy.py                  # Shannon entropy computation
│   ├── thresholding.py             # Otsu thresholding
│   ├── dct_compression.py          # DCT + quantization for ROI
│   ├── autoencoder.py              # Dense + Conv autoencoder for Non-ROI
│   ├── metrics.py                  # PSNR, SSIM, CR, Space Saving
│   ├── visualization.py            # Comparison plots, entropy histograms
│   └── reconstruction.py           # Full pipeline orchestration
├── main.py                         # CLI entry point (single image)
├── batch_evaluate.py               # Batch processing (entire folder)
├── hybrid_image_compression.py     # Original monolithic implementation
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Single Image (uses generated example if no image provided)

```bash
python main.py
python main.py path/to/image.png
python main.py image.png --block-size 16 --latent-dim 32 --epochs 100
```

### Choose Autoencoder Architecture

```bash
python main.py image.png --model dense    # Default: improved deep dense autoencoder
python main.py image.png --model conv     # Convolutional autoencoder
```

### Batch Processing

```bash
python batch_evaluate.py path/to/image_folder/
python batch_evaluate.py data/ --model conv --output-dir batch_outputs
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--block-size` | 16 | Block size for division |
| `--latent-dim` | 32 | Autoencoder bottleneck dimension |
| `--epochs` | 100 | Max training epochs |
| `--batch-size` | 16 | Training batch size |
| `--model` | dense | Autoencoder type: `dense` or `conv` |
| `--sparsity-weight` | 1e-4 | L1 sparsity regularization |
| `--output-dir` | outputs | Output directory |

## Outputs

| File | Description |
|------|-------------|
| `reconstructed_image.png` | Reconstructed image after compression |
| `roi_map.png` | Binary map showing ROI (white) and Non-ROI (black) blocks |
| `comparison.png` | 4-panel: original, ROI map, reconstructed, entropy histogram |
| `entropy_histogram.png` | Block entropy distribution with Otsu threshold |
| `results_summary.csv` | Batch mode: metrics for all images |

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **PSNR** | `20·log₁₀(255/√MSE)` | Higher = better quality (typical: 25-40 dB) |
| **SSIM** | Structural similarity | 1.0 = identical, higher = better |
| **CR** | `original_bits / compressed_bits` | Higher = more compression |
| **SS** | `(1 - 1/CR) × 100%` | Percentage of space saved |

## Datasets

- **Standard images**: Lena, Peppers, Cameraman, Barbara
- **Kodak dataset**: 24 images at 768×512 — http://r0k.us/graphics/kodak/
- **Medical MRI**: BrainWeb, OASIS

## Improvements Over Original

- **Deeper autoencoder**: 4-layer encoder/decoder with BatchNorm and Dropout
- **Convolutional option**: `--model conv` for spatial-aware compression
- **Early stopping**: Prevents overfitting during autoencoder training
- **Space Saving metric**: Added SS = (1 - 1/CR) × 100%
- **Entropy histogram**: Visual distribution of block entropies with Otsu threshold
- **Batch processing**: Process entire image folders with CSV summary

## Future Enhancements

- Replace autoencoder with **VAE** for smoother latent spaces
- Use **U-Net** for learned ROI segmentation instead of Otsu
- **Adaptive block sizes** via quad-tree decomposition
- **Learned quantization** matrices optimized end-to-end
