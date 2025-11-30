"""Krippendorff's Alpha computations."""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import krippendorff

from .models import AlphaResult, CoderData, CoderImpact, PairwiseOverall, PairwiseResult


def compute_alpha(
    data: np.ndarray,
    level: str = "nominal",
    value_domain: Optional[List] = None,
) -> float:
    """Compute Krippendorff's Alpha for a reliability matrix.

    Args:
        data: 2D array of shape (n_coders, n_units) with NaN for missing
        level: Measurement level ("nominal", "ordinal", "interval", "ratio")
        value_domain: Optional list of valid values

    Returns:
        Alpha value (float)
    """
    # Compute value domain from data if not provided
    if value_domain is None:
        flat = data[np.isfinite(data)]
        if len(flat) == 0:
            return np.nan
        value_domain = sorted(pd.unique(flat))

    try:
        alpha = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement=level,
            value_domain=value_domain if value_domain else None,
        )
        return float(alpha)
    except Exception:
        return np.nan


def compute_per_variable_alpha(
    coder_data: CoderData,
) -> List[AlphaResult]:
    """Compute Krippendorff's Alpha for each variable.

    Also computes alpha with each coder removed to assess coder impact.

    Args:
        coder_data: CoderData instance

    Returns:
        List of AlphaResult for each variable
    """
    results = []

    for var in coder_data.variables:
        level = coder_data.variable_levels[var]
        matrix = coder_data.get_reliability_matrix(var)

        # Alpha with all coders
        alpha_all = compute_alpha(matrix, level)

        # Alpha without each coder
        alpha_without = {}
        for coder_idx, coder in enumerate(coder_data.coder_names):
            # Create matrix without this coder
            mask = np.ones(len(coder_data.coder_names), dtype=bool)
            mask[coder_idx] = False
            subset_matrix = matrix[mask]
            alpha_without[coder] = compute_alpha(subset_matrix, level)

        # Find which coder's removal helps most
        gains = {coder: alpha_without[coder] - alpha_all for coder in coder_data.coder_names}
        if gains:
            best_coder = max(gains.items(), key=lambda x: x[1])
            max_gain = best_coder[1]
            coder_best = best_coder[0]
        else:
            max_gain = 0.0
            coder_best = ""

        results.append(AlphaResult(
            variable=var,
            level=level,
            alpha_all_coders=alpha_all,
            alpha_without=alpha_without,
            max_gain_if_removed=max_gain,
            coder_whose_removal_helps_most=coder_best,
        ))

    return results


def compute_pairwise_alpha(
    coder_data: CoderData,
) -> List[PairwiseResult]:
    """Compute pairwise Krippendorff's Alpha between each coder pair.

    Args:
        coder_data: CoderData instance

    Returns:
        List of PairwiseResult for each variable/coder pair combination
    """
    results = []

    for var in coder_data.variables:
        level = coder_data.variable_levels[var]
        matrix = coder_data.get_reliability_matrix(var)

        for coder_a, coder_b in itertools.combinations(coder_data.coder_names, 2):
            idx_a = coder_data.coder_names.index(coder_a)
            idx_b = coder_data.coder_names.index(coder_b)

            pair_matrix = np.array([matrix[idx_a], matrix[idx_b]])
            alpha_pair = compute_alpha(pair_matrix, level)

            results.append(PairwiseResult(
                variable=var,
                level=level,
                coder_a=coder_a,
                coder_b=coder_b,
                alpha_pair=alpha_pair,
            ))

    return results


def compute_pairwise_overall(
    pairwise_results: List[PairwiseResult],
) -> List[PairwiseOverall]:
    """Compute summary statistics for each coder pair across all variables.

    Args:
        pairwise_results: List of PairwiseResult from compute_pairwise_alpha

    Returns:
        List of PairwiseOverall summaries
    """
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame([r.to_dict() for r in pairwise_results])

    results = []
    for (coder_a, coder_b), group in df.groupby(['coder_a', 'coder_b']):
        alphas = group['alpha_pair'].dropna()

        results.append(PairwiseOverall(
            coder_a=coder_a,
            coder_b=coder_b,
            mean_alpha=alphas.mean() if len(alphas) > 0 else np.nan,
            median_alpha=alphas.median() if len(alphas) > 0 else np.nan,
            min_alpha=alphas.min() if len(alphas) > 0 else np.nan,
            max_alpha=alphas.max() if len(alphas) > 0 else np.nan,
            std_alpha=alphas.std(ddof=1) if len(alphas) > 1 else 0.0,
        ))

    return results


def compute_coder_impact(
    per_variable_results: List[AlphaResult],
    coder_names: List[str],
) -> List[CoderImpact]:
    """Analyze each coder's impact on reliability.

    Args:
        per_variable_results: Results from compute_per_variable_alpha
        coder_names: List of coder names

    Returns:
        List of CoderImpact for each coder
    """
    results = []

    for coder in coder_names:
        deltas = []
        for r in per_variable_results:
            if coder in r.alpha_without:
                delta = r.alpha_without[coder] - r.alpha_all_coders
                deltas.append(delta)

        deltas_arr = np.array(deltas)

        results.append(CoderImpact(
            coder=coder,
            mean_delta_when_removed=np.mean(deltas_arr) if len(deltas_arr) > 0 else 0.0,
            median_delta_when_removed=np.median(deltas_arr) if len(deltas_arr) > 0 else 0.0,
            num_vars_removal_increases_alpha=int((deltas_arr > 0).sum()),
            num_vars_removal_decreases_alpha=int((deltas_arr < 0).sum()),
            largest_increase_from_removal=float(deltas_arr.max()) if len(deltas_arr) > 0 else 0.0,
            largest_decrease_from_removal=float(deltas_arr.min()) if len(deltas_arr) > 0 else 0.0,
        ))

    # Sort by mean delta (most problematic first)
    results.sort(key=lambda x: x.mean_delta_when_removed, reverse=True)

    return results


def compute_overall_alpha(
    per_variable_results: List[AlphaResult],
    aggregation: str = "mean",
) -> float:
    """Compute overall alpha aggregated across all variables.

    Args:
        per_variable_results: Results from compute_per_variable_alpha
        aggregation: "mean" or "median"

    Returns:
        Overall alpha value
    """
    alphas = [r.alpha_all_coders for r in per_variable_results if not np.isnan(r.alpha_all_coders)]

    if not alphas:
        return np.nan

    if aggregation == "mean":
        return np.mean(alphas)
    elif aggregation == "median":
        return np.median(alphas)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_combo_overall(
    coder_data: CoderData,
    combo_sizes: List[int],
) -> pd.DataFrame:
    """Compute overall statistics for different coder combinations.

    Args:
        coder_data: CoderData instance
        combo_sizes: List of combination sizes to compute (e.g., [2, 3, 4])

    Returns:
        DataFrame with combination statistics
    """
    records = []

    for size in combo_sizes:
        for combo in itertools.combinations(coder_data.coder_names, size):
            alphas = []

            for var in coder_data.variables:
                level = coder_data.variable_levels[var]
                matrix = coder_data.get_reliability_matrix(var)

                # Get indices for this combination
                indices = [coder_data.coder_names.index(c) for c in combo]
                combo_matrix = matrix[indices]

                alpha = compute_alpha(combo_matrix, level)
                alphas.append(alpha)

            alphas_arr = np.array(alphas)
            alphas_valid = alphas_arr[~np.isnan(alphas_arr)]

            records.append({
                "coders": "+".join(combo),
                "n_coders": size,
                "mean_alpha": np.mean(alphas_valid) if len(alphas_valid) > 0 else np.nan,
                "median_alpha": np.median(alphas_valid) if len(alphas_valid) > 0 else np.nan,
                "min_alpha": np.min(alphas_valid) if len(alphas_valid) > 0 else np.nan,
                "max_alpha": np.max(alphas_valid) if len(alphas_valid) > 0 else np.nan,
                "std_alpha": np.std(alphas_valid, ddof=1) if len(alphas_valid) > 1 else 0.0,
            })

    return pd.DataFrame(records)


def interpret_alpha(alpha: float) -> Tuple[str, str]:
    """Interpret an alpha value according to standard thresholds.

    Args:
        alpha: Krippendorff's alpha value

    Returns:
        Tuple of (interpretation label, color for display)
    """
    if np.isnan(alpha):
        return "Cannot compute", "gray"
    elif alpha >= 0.80:
        return "Acceptable", "green"
    elif alpha >= 0.67:
        return "Tentative", "orange"
    else:
        return "Insufficient", "red"


def results_summary(
    per_variable_results: List[AlphaResult],
    pairwise_overall: List[PairwiseOverall],
    coder_impact: List[CoderImpact],
) -> str:
    """Generate a text summary of all results.

    Args:
        per_variable_results: Per-variable alpha results
        pairwise_overall: Pairwise summary results
        coder_impact: Coder impact results

    Returns:
        Summary text for LLM context
    """
    overall = compute_overall_alpha(per_variable_results)

    lines = [
        f"OVERALL RELIABILITY: {overall:.3f} ({interpret_alpha(overall)[0]})",
        "",
        "PER-VARIABLE ALPHA:",
    ]

    for r in per_variable_results:
        interp, _ = interpret_alpha(r.alpha_all_coders)
        lines.append(f"  - {r.variable}: {r.alpha_all_coders:.3f} ({interp})")

    lines.append("")
    lines.append("PAIRWISE CODER AGREEMENT (mean alpha):")

    for p in pairwise_overall:
        lines.append(f"  - {p.coder_a} vs {p.coder_b}: {p.mean_alpha:.3f}")

    lines.append("")
    lines.append("CODER IMPACT (delta when removed):")

    for c in coder_impact:
        impact = "improves" if c.mean_delta_when_removed > 0 else "reduces"
        lines.append(f"  - {c.coder}: {c.mean_delta_when_removed:+.3f} ({impact} reliability)")

    # Identify problematic variables
    low_alpha_vars = [r for r in per_variable_results if r.alpha_all_coders < 0.67]
    if low_alpha_vars:
        lines.append("")
        lines.append("VARIABLES NEEDING ATTENTION (alpha < 0.67):")
        for r in low_alpha_vars:
            lines.append(f"  - {r.variable}: {r.alpha_all_coders:.3f}")

    return "\n".join(lines)
