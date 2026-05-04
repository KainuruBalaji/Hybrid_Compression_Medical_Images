"""Module 2: Entropy Calculation — Shannon entropy per block + normalization."""

from typing import Tuple

import numpy as np


def compute_block_entropy(block: np.ndarray) -> float:
    """Compute Shannon entropy for a single image block.

    H(B) = -Σ p(i) · log₂(p(i))  for all i where p(i) > 0

    Higher entropy indicates more complex texture/detail.
    Lower entropy indicates smooth/uniform regions.
    """
    histogram = np.bincount(block.flatten(), minlength=256).astype(np.float64)
    probabilities = histogram / np.sum(histogram)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def entropy_calculation(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and normalized entropy for all blocks.

    Raw entropy: H(B_k) for each block k
    Normalized: H_norm(B_k) = H(B_k) / Σ H(B_j)

    Returns:
        entropies: raw entropy values per block
        normalized: normalized entropy values (sum to 1.0)
    """
    entropies = np.array(
        [compute_block_entropy(block) for block in blocks],
        dtype=np.float32,
    )

    total_entropy = float(np.sum(entropies))
    if total_entropy > 0:
        normalized = entropies / total_entropy
    else:
        normalized = np.zeros_like(entropies)

    return entropies, normalized
