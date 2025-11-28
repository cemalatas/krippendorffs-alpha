"""Additional reliability statistics: Cohen's Kappa, Scott's Pi, Percent Agreement."""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .models import AdditionalStats, CoderData


def compute_percent_agreement(
    coder_data: CoderData,
    variable: str,
) -> Dict[str, float]:
    """Compute percent agreement statistics for a variable.

    Args:
        coder_data: CoderData instance
        variable: Variable name

    Returns:
        Dict with:
            - overall_agreement: % of units where ALL coders agree
            - pairwise_mean: Mean pairwise agreement
            - by_pair: Dict of agreement for each coder pair
    """
    matrix = coder_data.get_reliability_matrix(variable)
    n_coders, n_units = matrix.shape

    # Overall: all coders agree
    def all_agree(col):
        valid = col[~np.isnan(col)]
        return len(np.unique(valid)) <= 1 if len(valid) > 0 else True

    overall = np.mean([all_agree(matrix[:, i]) for i in range(n_units)])

    # Pairwise agreements
    pairs = {}
    for coder_a, coder_b in itertools.combinations(coder_data.coder_names, 2):
        arr_a, arr_b = coder_data.get_pairwise_data(variable, coder_a, coder_b)
        valid = ~np.isnan(arr_a) & ~np.isnan(arr_b)

        if valid.sum() > 0:
            agreement = np.mean(arr_a[valid] == arr_b[valid])
        else:
            agreement = np.nan

        pairs[f"{coder_a}-{coder_b}"] = agreement

    pairwise_values = [v for v in pairs.values() if not np.isnan(v)]

    return {
        "overall_agreement": overall,
        "pairwise_mean": np.mean(pairwise_values) if pairwise_values else np.nan,
        "by_pair": pairs,
    }


def compute_cohens_kappa(
    coder_data: CoderData,
    variable: str,
    coder_a: str,
    coder_b: str,
    weights: Optional[str] = None,
) -> float:
    """Compute Cohen's Kappa between two coders.

    Args:
        coder_data: CoderData instance
        variable: Variable name
        coder_a: First coder name
        coder_b: Second coder name
        weights: None for nominal, "linear" or "quadratic" for ordinal

    Returns:
        Cohen's Kappa value
    """
    arr_a, arr_b = coder_data.get_pairwise_data(variable, coder_a, coder_b)
    valid_mask = ~np.isnan(arr_a) & ~np.isnan(arr_b)

    if valid_mask.sum() < 2:
        return np.nan

    a = arr_a[valid_mask].astype(int)
    b = arr_b[valid_mask].astype(int)

    # Auto-select weights based on measurement level
    level = coder_data.variable_levels.get(variable, "nominal")
    if weights is None:
        if level == "ordinal":
            weights = "linear"
        elif level in ("interval", "ratio"):
            weights = "quadratic"
        else:
            weights = None

    try:
        return cohen_kappa_score(a, b, weights=weights)
    except Exception:
        return np.nan


def compute_pairwise_kappa(
    coder_data: CoderData,
    variable: str,
) -> Dict[str, float]:
    """Compute Cohen's Kappa for all coder pairs.

    Args:
        coder_data: CoderData instance
        variable: Variable name

    Returns:
        Dict mapping "CoderA-CoderB" to kappa value
    """
    kappas = {}

    for coder_a, coder_b in itertools.combinations(coder_data.coder_names, 2):
        kappa = compute_cohens_kappa(coder_data, variable, coder_a, coder_b)
        kappas[f"{coder_a}-{coder_b}"] = kappa

    return kappas


def compute_scotts_pi(
    coder_data: CoderData,
    variable: str,
    coder_a: str,
    coder_b: str,
) -> float:
    """Compute Scott's Pi between two coders.

    Scott's Pi uses pooled marginal distributions (unlike Cohen's Kappa).

    Args:
        coder_data: CoderData instance
        variable: Variable name
        coder_a: First coder name
        coder_b: Second coder name

    Returns:
        Scott's Pi value
    """
    arr_a, arr_b = coder_data.get_pairwise_data(variable, coder_a, coder_b)
    valid_mask = ~np.isnan(arr_a) & ~np.isnan(arr_b)

    if valid_mask.sum() < 2:
        return np.nan

    a = arr_a[valid_mask]
    b = arr_b[valid_mask]
    n = len(a)

    # Observed agreement
    P_o = np.mean(a == b)

    # Expected agreement using pooled marginals
    all_values = np.concatenate([a, b])
    categories = np.unique(all_values)

    P_e = 0.0
    for cat in categories:
        prop = np.sum(all_values == cat) / (2 * n)
        P_e += prop ** 2

    if P_e >= 1.0:
        return 1.0 if P_o == 1.0 else 0.0

    return (P_o - P_e) / (1 - P_e)


def compute_fleiss_kappa(
    coder_data: CoderData,
    variable: str,
) -> float:
    """Compute Fleiss' Kappa for multiple coders.

    Fleiss' Kappa is the multi-coder generalization of Scott's Pi.

    Args:
        coder_data: CoderData instance
        variable: Variable name

    Returns:
        Fleiss' Kappa value
    """
    matrix = coder_data.get_reliability_matrix(variable)
    n_coders, n_units = matrix.shape

    # Get all unique categories
    flat = matrix[np.isfinite(matrix)]
    if len(flat) == 0:
        return np.nan

    categories = sorted(np.unique(flat))
    n_categories = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Build count matrix: (n_units, n_categories)
    # counts[i, j] = number of coders who assigned category j to unit i
    counts = np.zeros((n_units, n_categories))

    for unit_idx in range(n_units):
        for coder_idx in range(n_coders):
            val = matrix[coder_idx, unit_idx]
            if not np.isnan(val) and val in cat_to_idx:
                counts[unit_idx, cat_to_idx[val]] += 1

    # Number of coders per unit
    n_per_unit = counts.sum(axis=1)

    # Filter out units with fewer than 2 coders
    valid_units = n_per_unit >= 2
    if valid_units.sum() == 0:
        return np.nan

    counts = counts[valid_units]
    n_per_unit = n_per_unit[valid_units]
    n_valid = len(counts)

    # Compute P_i (agreement for each unit)
    # P_i = (1 / (n_i * (n_i - 1))) * sum_j(n_ij * (n_ij - 1))
    P_i = np.zeros(n_valid)
    for i in range(n_valid):
        n_i = n_per_unit[i]
        if n_i > 1:
            sum_nij_squared = np.sum(counts[i] * (counts[i] - 1))
            P_i[i] = sum_nij_squared / (n_i * (n_i - 1))

    # Mean observed agreement
    P_bar = np.mean(P_i)

    # Category proportions (p_j)
    total_ratings = counts.sum()
    p_j = counts.sum(axis=0) / total_ratings

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    if P_e >= 1.0:
        return 1.0 if P_bar == 1.0 else 0.0

    return (P_bar - P_e) / (1 - P_e)


def compute_all_additional_stats(
    coder_data: CoderData,
    variable: str,
) -> AdditionalStats:
    """Compute all additional statistics for a variable.

    Args:
        coder_data: CoderData instance
        variable: Variable name

    Returns:
        AdditionalStats instance
    """
    # Percent agreement
    agreement = compute_percent_agreement(coder_data, variable)

    # Cohen's Kappa (pairwise)
    kappas = compute_pairwise_kappa(coder_data, variable)

    # Fleiss' Kappa (multi-coder)
    fleiss = compute_fleiss_kappa(coder_data, variable)

    return AdditionalStats(
        variable=variable,
        percent_agreement_overall=agreement["overall_agreement"],
        percent_agreement_pairwise=agreement["by_pair"],
        cohens_kappa=kappas,
        fleiss_kappa=fleiss,
    )


def compute_confusion_matrix(
    coder_data: CoderData,
    variable: str,
    coder_a: str,
    coder_b: str,
) -> Tuple[np.ndarray, List]:
    """Compute confusion matrix between two coders.

    Args:
        coder_data: CoderData instance
        variable: Variable name
        coder_a: First coder name
        coder_b: Second coder name

    Returns:
        Tuple of (confusion matrix, category labels)
    """
    arr_a, arr_b = coder_data.get_pairwise_data(variable, coder_a, coder_b)
    valid = ~np.isnan(arr_a) & ~np.isnan(arr_b)

    a = arr_a[valid]
    b = arr_b[valid]

    # Get unique categories
    categories = sorted(np.unique(np.concatenate([a, b])))
    n_cats = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Build confusion matrix
    conf_matrix = np.zeros((n_cats, n_cats), dtype=int)
    for val_a, val_b in zip(a, b):
        i = cat_to_idx[val_a]
        j = cat_to_idx[val_b]
        conf_matrix[i, j] += 1

    return conf_matrix, categories


def stats_summary_for_variable(
    coder_data: CoderData,
    variable: str,
    alpha: float,
) -> str:
    """Generate a summary of all statistics for a variable.

    Args:
        coder_data: CoderData instance
        variable: Variable name
        alpha: Krippendorff's alpha for this variable

    Returns:
        Summary text
    """
    stats = compute_all_additional_stats(coder_data, variable)

    lines = [
        f"Statistics for '{variable}':",
        f"  Krippendorff's Alpha: {alpha:.3f}",
        f"  Fleiss' Kappa: {stats.fleiss_kappa:.3f}" if stats.fleiss_kappa else "",
        f"  Overall Agreement: {stats.percent_agreement_overall:.1%}",
        "",
        "  Pairwise Statistics:",
    ]

    for pair, agree in stats.percent_agreement_pairwise.items():
        kappa = stats.cohens_kappa.get(pair, np.nan) if stats.cohens_kappa else np.nan
        lines.append(f"    {pair}: {agree:.1%} agreement, κ={kappa:.3f}")

    return "\n".join([l for l in lines if l])
