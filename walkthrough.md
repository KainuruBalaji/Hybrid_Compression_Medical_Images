# DCTAE Implementation Walkthrough

## What Was Built

A complete modular implementation of the **"Hybrid Image Compression Using DCT and Autoencoder"** paper, restructured from a monolithic 423-line script into an 8-module Python package with CLI tools.

## Project Structure

```
dctae/
├── __init__.py           # Package exports
├── preprocessing.py      # Image loading, padding, block division, merge
├── entropy.py            # Shannon entropy per block + normalization
├── thresholding.py       # Otsu threshold for ROI/Non-ROI classification
├── dct_compression.py    # 2D DCT + JPEG quantization for ROI blocks
├── autoencoder.py        # Improved dense + convolutional autoencoder
├── metrics.py            # PSNR, SSIM, Compression Ratio, Space Saving
├── visualization.py      # 4-panel comparison + entropy histogram
└── reconstruction.py     # Full pipeline orchestration
main.py                   # Single-image CLI entry point
batch_evaluate.py         # Batch processing with CSV summary
```

## Key Improvements Over Original

| Aspect | Before | After |
|--------|--------|-------|
| Autoencoder depth | 2 layers (256→128→32→128→256) | 4 layers (256→192→128→64→32→64→128→192→256) |
| Regularization | None | BatchNorm + Dropout(0.1) |
| Training | Fixed 40 epochs | Up to 100 with EarlyStopping |
| Conv AE option | ❌ | ✅ `--model conv` |
| Space Saving metric | ❌ | ✅ |
| Entropy histogram | ❌ | ✅ 4-panel viz + standalone |
| Batch processing | ❌ | ✅ with CSV summary |
| Visualization | 3-panel (1×3) | 4-panel (2×2) with entropy plot |

## Test Results

### Example MRI Image (256×256)
```
Entropy threshold  : 5.0670
ROI blocks         : 91
Non-ROI blocks     : 165
PSNR               : 29.6271 dB
SSIM               : 0.6598
Compression Ratio  : 6.6846
Space Saving       : 85.04%
```

### Tiger Image (real photo)
```
Entropy threshold  : 4.2718
ROI blocks         : 496
Non-ROI blocks     : 384
PSNR               : 27.1983 dB
SSIM               : 0.8242
Compression Ratio  : 3.6182
Space Saving       : 72.36%
```

## Visual Outputs

### Example Image — 4-Panel Comparison
![Example image comparison](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/outputs_new/comparison.png)

### Tiger Image — 4-Panel Comparison
![Tiger image comparison](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/outputs_tiger/comparison.png)

### Entropy Histogram with Otsu Threshold
![Entropy histogram](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/outputs_tiger/entropy_histogram.png)

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| [preprocessing.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/preprocessing.py) | NEW | Image loading, padding, block division |
| [entropy.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/entropy.py) | NEW | Shannon entropy computation |
| [thresholding.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/thresholding.py) | NEW | Otsu thresholding |
| [dct_compression.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/dct_compression.py) | NEW | DCT + JPEG quantization |
| [autoencoder.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/autoencoder.py) | NEW | Improved dense + conv autoencoder |
| [metrics.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/metrics.py) | NEW | PSNR, SSIM, CR, Space Saving |
| [visualization.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/visualization.py) | NEW | 4-panel comparison + entropy histogram |
| [reconstruction.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/reconstruction.py) | NEW | Full pipeline orchestration |
| [__init__.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/dctae/__init__.py) | NEW | Package exports |
| [main.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/main.py) | NEW | CLI entry point |
| [batch_evaluate.py](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/batch_evaluate.py) | NEW | Batch processing script |
| [requirements.txt](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/requirements.txt) | MODIFIED | Added scipy |
| [README.md](file:///c:/Users/CSLAB/Downloads/MS_Project/MS_Project/README.md) | MODIFIED | Full documentation rewrite |

> [!NOTE]
> The original `hybrid_image_compression.py` is preserved unchanged for reference.
