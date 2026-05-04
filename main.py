

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

from dctae import reconstruction_pipeline, create_example_image


def main() -> None:
    # CLI entry point that handles arguments and runs the pipeline for a single image.
    parser = argparse.ArgumentParser(
        description="DCTAE: Hybrid Image Compression Using DCT and Autoencoder"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to input image. If omitted, an example image is generated.",
    )
    parser.add_argument("--block-size", type=int, default=16,
                        help="Non-overlapping block size (default: 16)")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Autoencoder latent dimension (default: 32)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max autoencoder training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--model", type=str, default="dense",
                        choices=["dense", "conv"],
                        help="Autoencoder type: 'dense' or 'conv' (default: dense)")
    parser.add_argument("--sparsity-weight", type=float, default=1e-4,
                        help="L1 sparsity regularization weight (default: 1e-4)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    args = parser.parse_args()

    if args.image_path is None:
        example_path = "example_input.png"
        if not Path(example_path).exists():
            create_example_image(example_path)
        input_path = example_path
    else:
        input_path = args.image_path
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

    results = reconstruction_pipeline(
        image_path=input_path,
        block_size=args.block_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model,
        sparsity_weight=args.sparsity_weight,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("  DCTAE: Hybrid Image Compression (DCT + Autoencoder)")
    print("=" * 60)
    print(f"  Input image        : {input_path}")
    print(f"  Block size         : {args.block_size}x{args.block_size}")
    print(f"  Model type         : {args.model}")
    print(f"  Latent dim         : {args.latent_dim}")
    print("-" * 60)
    print(f"  Entropy threshold  : {results['threshold']:.4f}")
    print(f"  ROI blocks         : {results['roi_block_count']}")
    print(f"  Non-ROI blocks     : {results['non_roi_block_count']}")
    print(f"  Total blocks       : {results['total_blocks']}")
    print("-" * 60)
    print(f"  PSNR               : {results['psnr']:.4f} dB")
    print(f"  SSIM               : {results['ssim']:.4f}")
    print(f"  Compression Ratio  : {results['compression_ratio']:.4f}")
    print(f"  Space Saving       : {results['space_saving']:.2f}%")
    print(f"  BPP                : {results['bpp']:.4f}")
    print("-" * 60)
    print(f"  skimage available  : {results['skimage_available']}")
    print(f"  Outputs saved to   : {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
