

from typing import Tuple

import numpy as np


def compute_block_entropy(block: np.ndarray) -> float:
    # Calculates the Shannon entropy for a single image block.
    histogram = np.bincount(block.flatten(), minlength=256).astype(np.float64)
    probabilities = histogram / np.sum(histogram)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def entropy_calculation(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Computes raw and normalized entropies for a list of blocks.
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
