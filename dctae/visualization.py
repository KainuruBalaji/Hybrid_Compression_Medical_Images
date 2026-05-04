"""Module 7: Visualization — comparison plots and entropy histogram."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_histogram(
    entropies: np.ndarray,
    threshold: float,
    output_path: str,
) -> None:
    """Plot histogram of block entropies with the Otsu threshold line.

    This visualization shows the bimodal distribution of entropy values
    and where Otsu's method splits ROI from Non-ROI blocks.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    roi_mask = entropies > threshold
    non_roi_mask = ~roi_mask

    bins = np.linspace(float(np.min(entropies)) - 0.1, float(np.max(entropies)) + 0.1, 40)

    if np.any(non_roi_mask):
        ax.hist(entropies[non_roi_mask], bins=bins, alpha=0.7,
                color="#3498db", label=f"Non-ROI ({int(np.sum(non_roi_mask))} blocks)", edgecolor="white")
    if np.any(roi_mask):
        ax.hist(entropies[roi_mask], bins=bins, alpha=0.7,
                color="#e74c3c", label=f"ROI ({int(np.sum(roi_mask))} blocks)", edgecolor="white")

    ax.axvline(threshold, color="#2c3e50", linewidth=2, linestyle="--",
               label=f"Otsu threshold = {threshold:.3f}")

    ax.set_xlabel("Shannon Entropy (bits)", fontsize=12)
    ax.set_ylabel("Number of Blocks", fontsize=12)
    ax.set_title("Block Entropy Distribution with Otsu Threshold", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_results(
    original: np.ndarray,
    reconstructed: np.ndarray,
    roi_map: np.ndarray,
    entropies: np.ndarray,
    threshold: float,
    output_path: str,
) -> None:
    """Save 4-panel visualization: original, ROI map, reconstructed, entropy histogram."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(roi_map, cmap="gray")
    axes[0, 1].set_title("ROI (white) / Non-ROI (black) Map", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(reconstructed, cmap="gray")
    axes[1, 0].set_title("Reconstructed Image", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    roi_mask = entropies > threshold
    non_roi_mask = ~roi_mask
    bins = np.linspace(float(np.min(entropies)) - 0.1, float(np.max(entropies)) + 0.1, 30)
    if np.any(non_roi_mask):
        axes[1, 1].hist(entropies[non_roi_mask], bins=bins, alpha=0.7,
                        color="#3498db", label="Non-ROI", edgecolor="white")
    if np.any(roi_mask):
        axes[1, 1].hist(entropies[roi_mask], bins=bins, alpha=0.7,
                        color="#e74c3c", label="ROI", edgecolor="white")
    axes[1, 1].axvline(threshold, color="#2c3e50", linewidth=2, linestyle="--",
                       label=f"Otsu T={threshold:.2f}")
    axes[1, 1].set_xlabel("Entropy (bits)")
    axes[1, 1].set_ylabel("Block Count")
    axes[1, 1].set_title("Entropy Distribution", fontsize=13, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("DCTAE: Hybrid Image Compression (DCT + Autoencoder)", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_simple(
    original: np.ndarray,
    reconstructed: np.ndarray,
    roi_map: np.ndarray,
    output_path: str,
) -> None:
    """Save a simple 3-panel comparison (original, ROI map, reconstructed)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(roi_map, cmap="gray")
    axes[1].set_title("ROI / Non-ROI Map")
    axes[2].imshow(reconstructed, cmap="gray")
    axes[2].set_title("Reconstructed")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
