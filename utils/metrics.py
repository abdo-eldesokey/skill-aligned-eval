import numpy as np
import pandas as pd
from collections import defaultdict


def krippendorff_alpha(data: np.ndarray, level: str = "nominal") -> float:
    """
    Calculate Krippendorff's Alpha for inter-annotator agreement.
    
    Args:
        data: 2D array where rows are items and columns are annotators.
              Missing values should be np.nan.
        level: 'nominal', 'ordinal', or 'interval'
    
    Returns:
        Alpha coefficient (-1 to 1, where 1 is perfect agreement)
    """
    # Remove items with less than 2 annotators
    valid_mask = np.sum(~np.isnan(data), axis=1) >= 2
    data = data[valid_mask]
    
    if data.shape[0] == 0:
        return np.nan
    
    # Get all unique values (excluding NaN)
    all_values = data[~np.isnan(data)]
    if len(all_values) == 0:
        return np.nan
    
    unique_values = np.unique(all_values)
    n_values = len(unique_values)
    
    if n_values <= 1:
        return 1.0  # Perfect agreement if only one value
    
    # Create value to index mapping
    value_to_idx = {v: i for i, v in enumerate(unique_values)}
    
    # Calculate observed disagreement
    n_items, n_annotators = data.shape
    
    # Count pairs within each item
    observed_pairs = defaultdict(int)
    expected_pairs = defaultdict(int)
    total_pairs = 0
    
    # Marginal counts
    marginals = np.zeros(n_values)
    
    for item in data:
        valid = item[~np.isnan(item)]
        n_valid = len(valid)
        if n_valid < 2:
            continue
        
        # Count pairs in this item
        for i in range(n_valid):
            vi = value_to_idx[valid[i]]
            marginals[vi] += 1
            for j in range(i + 1, n_valid):
                vj = value_to_idx[valid[j]]
                observed_pairs[(vi, vj)] += 1
                observed_pairs[(vj, vi)] += 1
                total_pairs += 2
    
    if total_pairs == 0:
        return np.nan
    
    # Calculate disagreement based on level
    def metric(vi, vj):
        if level == "nominal":
            return 0 if vi == vj else 1
        elif level == "ordinal":
            return (vi - vj) ** 2
        else:  # interval
            return (unique_values[vi] - unique_values[vj]) ** 2
    
    # Observed disagreement
    Do = 0
    for (vi, vj), count in observed_pairs.items():
        Do += count * metric(vi, vj)
    Do /= total_pairs
    
    # Expected disagreement
    total_annotations = np.sum(marginals)
    De = 0
    for vi in range(n_values):
        for vj in range(n_values):
            De += marginals[vi] * marginals[vj] * metric(vi, vj)
    De /= (total_annotations * (total_annotations - 1))
    
    if De == 0:
        return 1.0
    
    alpha = 1 - Do / De
    return alpha


def bootstrap_alpha(pivot_df, level: str = "nominal", n_boot: int = 1000,
                    ci: float = 0.95, seed: int = 0):
    """
    Bootstrap confidence interval for Krippendorff's alpha by resampling items
    (rows of the annotator-pivot) with replacement.

    Args:
        pivot_df: DataFrame or 2D ndarray with shape (n_items, n_annotators);
                  NaN denotes missing.
        level: 'nominal', 'ordinal', or 'interval'.
        n_boot: number of bootstrap resamples.
        ci: confidence level (e.g. 0.95 for 95% CI).
        seed: numpy RNG seed for reproducibility.

    Returns:
        (alpha, lo, hi): point estimate plus percentile CI bounds. NaN values
        are returned if the data is degenerate.
    """
    if isinstance(pivot_df, pd.DataFrame):
        values = pivot_df.values
    else:
        values = np.asarray(pivot_df)

    base = krippendorff_alpha(values, level=level)
    n = values.shape[0]
    if n == 0 or np.isnan(base):
        return base, np.nan, np.nan

    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = krippendorff_alpha(values[idx], level=level)
        if not np.isnan(a):
            alphas.append(a)

    if not alphas:
        return base, np.nan, np.nan

    lo_pct = 100 * (1 - ci) / 2
    hi_pct = 100 * (1 + ci) / 2
    lo, hi = np.percentile(alphas, [lo_pct, hi_pct])
    return base, float(lo), float(hi)