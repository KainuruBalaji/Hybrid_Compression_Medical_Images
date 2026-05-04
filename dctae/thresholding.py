

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    # Calculates Otsu's optimal threshold over block entropy values to separate ROI from Non-ROI.
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
