"""Batch evaluation — process all images in a directory and produce a CSV summary."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

from dctae import reconstruction_pipeline

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pgm"}


def batch_evaluate(
    input_dir: str,
    block_size: int = 16,
    latent_dim: int = 32,
    epochs: int = 100,
    batch_size: int = 16,
    model_type: str = "dense",
    output_dir: str = "batch_outputs",
) -> None:
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    image_files = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    csv_path = output_base / "results_summary.csv"
    fieldnames = [
        "filename", "roi_blocks", "non_roi_blocks", "total_blocks",
        "threshold", "psnr", "ssim", "compression_ratio", "space_saving", "bpp",
    ]

    all_results = []

    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")

        img_output_dir = str(output_base / image_file.stem)

        try:
            results = reconstruction_pipeline(
                image_path=str(image_file),
                block_size=block_size,
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size,
                model_type=model_type,
                output_dir=img_output_dir,
            )

            row = {
                "filename": image_file.name,
                "roi_blocks": results["roi_block_count"],
                "non_roi_blocks": results["non_roi_block_count"],
                "total_blocks": results["total_blocks"],
                "threshold": f"{results['threshold']:.4f}",
                "psnr": f"{results['psnr']:.4f}",
                "ssim": f"{results['ssim']:.4f}",
                "compression_ratio": f"{results['compression_ratio']:.4f}",
                "space_saving": f"{results['space_saving']:.2f}",
                "bpp": f"{results['bpp']:.4f}",
            }
            all_results.append(row)

            print(f"  PSNR={results['psnr']:.2f} dB  SSIM={results['ssim']:.4f}  "
                  f"CR={results['compression_ratio']:.2f}  SS={results['space_saving']:.1f}%  "
                  f"BPP={results['bpp']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"filename": image_file.name, "psnr": "ERROR"})

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {csv_path}")

    valid = [r for r in all_results if "ERROR" not in str(r.get("psnr", ""))]
    if valid:
        avg_psnr = sum(float(r["psnr"]) for r in valid) / len(valid)
        avg_ssim = sum(float(r["ssim"]) for r in valid) / len(valid)
        avg_cr = sum(float(r["compression_ratio"]) for r in valid) / len(valid)
        avg_ss = sum(float(r["space_saving"]) for r in valid) / len(valid)
        avg_bpp = sum(float(r["bpp"]) for r in valid) / len(valid)
        print(f"\nAverages over {len(valid)} images:")
        print(f"  PSNR: {avg_psnr:.4f} dB  |  SSIM: {avg_ssim:.4f}  |  CR: {avg_cr:.4f}  |  SS: {avg_ss:.2f}%  |  BPP: {avg_bpp:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate DCTAE on a folder of images")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default="dense", choices=["dense", "conv"])
    parser.add_argument("--output-dir", type=str, default="batch_outputs")
    args = parser.parse_args()

    batch_evaluate(
        input_dir=args.input_dir,
        block_size=args.block_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
