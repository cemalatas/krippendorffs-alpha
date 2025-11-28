"""Per-row disagreement analysis for intercoder reliability."""

import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import CoderData, Disagreement


def find_disagreements(
    coder_data: CoderData,
    variable: str,
) -> List[Disagreement]:
    """Find all units with coder disagreements for a variable.

    Args:
        coder_data: CoderData instance
        variable: Variable name

    Returns:
        List of Disagreement objects
    """
    matrix = coder_data.get_reliability_matrix(variable)
    disagreements = []

    for unit_idx, unit_id in enumerate(coder_data.unit_ids):
        values = matrix[:, unit_idx]
        valid_values = values[~np.isnan(values)]

        # Check if there's disagreement
        if len(valid_values) < 2:
            continue  # Can't have disagreement with < 2 values

        unique_values = np.unique(valid_values)
        if len(unique_values) <= 1:
            continue  # All coders agree

        # Build coder values dict
        coder_values = {}
        for coder_idx, coder in enumerate(coder_data.coder_names):
            val = values[coder_idx]
            if np.isnan(val):
                coder_values[coder] = None
            else:
                # Convert to int if it's a whole number
                coder_values[coder] = int(val) if val == int(val) else val

        # Determine disagreement type
        n_valid = len(valid_values)
        n_missing = len(values) - n_valid
        value_counts = Counter(valid_values)

        if n_missing > 0:
            dtype = "missing"
        elif len(value_counts) == 2 and min(value_counts.values()) == 1:
            dtype = "outlier"  # One coder differs from all others
        else:
            dtype = "split"  # Multiple distinct values

        # Compute severity (normalized disagreement)
        if len(unique_values) > 1:
            val_range = max(valid_values) - min(valid_values)
            max_possible = max(valid_values) if max(valid_values) != 0 else 1
            severity = val_range / max_possible
        else:
            severity = 0.0

        disagreements.append(Disagreement(
            unit_id=str(unit_id),
            variable=variable,
            coder_values=coder_values,
            disagreement_type=dtype,
            severity=min(severity, 1.0),
        ))

    return disagreements


def find_all_disagreements(
    coder_data: CoderData,
) -> List[Disagreement]:
    """Find all disagreements across all variables.

    Args:
        coder_data: CoderData instance

    Returns:
        List of all Disagreement objects
    """
    all_disagreements = []

    for var in coder_data.variables:
        disagreements = find_disagreements(coder_data, var)
        all_disagreements.extend(disagreements)

    return all_disagreements


def disagreement_summary(
    coder_data: CoderData,
) -> pd.DataFrame:
    """Generate summary statistics of disagreements across all variables.

    Args:
        coder_data: CoderData instance

    Returns:
        DataFrame with disagreement summary per variable
    """
    records = []

    for var in coder_data.variables:
        disagreements = find_disagreements(coder_data, var)
        n_units = len(coder_data.unit_ids)

        # Count disagreement types
        type_counts = Counter(d.disagreement_type for d in disagreements)

        # Find which coder pair disagrees most
        pair_counts = defaultdict(int)
        for d in disagreements:
            vals = {k: v for k, v in d.coder_values.items() if v is not None}
            if len(set(vals.values())) > 1:
                # Count each disagreeing pair
                for ca, cb in itertools.combinations(vals.keys(), 2):
                    if vals[ca] != vals[cb]:
                        pair_key = f"{ca}-{cb}"
                        pair_counts[pair_key] += 1

        most_disagreeing = (
            max(pair_counts.items(), key=lambda x: x[1])[0]
            if pair_counts else "N/A"
        )

        records.append({
            "variable": var,
            "n_disagreements": len(disagreements),
            "pct_disagreements": len(disagreements) / n_units * 100 if n_units > 0 else 0,
            "n_split": type_counts.get("split", 0),
            "n_outlier": type_counts.get("outlier", 0),
            "n_missing": type_counts.get("missing", 0),
            "mean_severity": np.mean([d.severity for d in disagreements]) if disagreements else 0,
            "coders_most_often_disagreeing": most_disagreeing,
        })

    return pd.DataFrame(records)


def get_disagreement_heatmap_data(
    coder_data: CoderData,
) -> pd.DataFrame:
    """Get data for disagreement heatmap (variables x coder pairs).

    Args:
        coder_data: CoderData instance

    Returns:
        DataFrame with rows=variables, columns=coder pairs, values=counts
    """
    pairs = list(itertools.combinations(coder_data.coder_names, 2))
    pair_labels = [f"{a}-{b}" for a, b in pairs]

    data = []
    for var in coder_data.variables:
        row = {"variable": var}

        disagreements = find_disagreements(coder_data, var)

        # Count disagreements per pair
        for pair_label, (ca, cb) in zip(pair_labels, pairs):
            count = 0
            for d in disagreements:
                val_a = d.coder_values.get(ca)
                val_b = d.coder_values.get(cb)
                if val_a is not None and val_b is not None and val_a != val_b:
                    count += 1
            row[pair_label] = count

        data.append(row)

    return pd.DataFrame(data).set_index("variable")


def filter_disagreements(
    disagreements: List[Disagreement],
    variable: Optional[str] = None,
    coder: Optional[str] = None,
    disagreement_type: Optional[str] = None,
    min_severity: float = 0.0,
) -> List[Disagreement]:
    """Filter disagreements by various criteria.

    Args:
        disagreements: List of Disagreement objects
        variable: Filter by variable name
        coder: Filter by coder involvement
        disagreement_type: Filter by type ("split", "outlier", "missing")
        min_severity: Minimum severity threshold

    Returns:
        Filtered list of Disagreement objects
    """
    filtered = disagreements

    if variable:
        filtered = [d for d in filtered if d.variable == variable]

    if coder:
        filtered = [d for d in filtered if coder in d.coder_values]

    if disagreement_type:
        filtered = [d for d in filtered if d.disagreement_type == disagreement_type]

    if min_severity > 0:
        filtered = [d for d in filtered if d.severity >= min_severity]

    return filtered


def get_unit_disagreement_detail(
    coder_data: CoderData,
    unit_id: str,
) -> Dict[str, Dict]:
    """Get detailed disagreement info for a specific unit.

    Args:
        coder_data: CoderData instance
        unit_id: Unit identifier

    Returns:
        Dict mapping variable -> {coder_values, has_disagreement, majority_value}
    """
    unit_idx = coder_data.unit_ids.index(unit_id)
    result = {}

    for var in coder_data.variables:
        matrix = coder_data.get_reliability_matrix(var)
        values = matrix[:, unit_idx]

        coder_values = {}
        for coder_idx, coder in enumerate(coder_data.coder_names):
            val = values[coder_idx]
            if np.isnan(val):
                coder_values[coder] = None
            else:
                coder_values[coder] = int(val) if val == int(val) else val

        valid_values = [v for v in coder_values.values() if v is not None]
        has_disagreement = len(set(valid_values)) > 1 if len(valid_values) >= 2 else False

        # Find majority value
        if valid_values:
            value_counts = Counter(valid_values)
            majority_value = value_counts.most_common(1)[0][0]
        else:
            majority_value = None

        result[var] = {
            "coder_values": coder_values,
            "has_disagreement": has_disagreement,
            "majority_value": majority_value,
        }

    return result


def compute_coder_disagreement_profile(
    coder_data: CoderData,
) -> pd.DataFrame:
    """Compute how often each coder disagrees with others.

    Args:
        coder_data: CoderData instance

    Returns:
        DataFrame with coder disagreement profile
    """
    all_disagreements = find_all_disagreements(coder_data)

    # Count disagreements involving each coder
    coder_counts = defaultdict(lambda: {"involved": 0, "as_outlier": 0})

    for d in all_disagreements:
        vals = {k: v for k, v in d.coder_values.items() if v is not None}

        for coder in vals.keys():
            coder_counts[coder]["involved"] += 1

        # Check if one coder is the outlier
        if d.disagreement_type == "outlier":
            value_counts = Counter(vals.values())
            outlier_value = value_counts.most_common()[-1][0]  # Least common
            for coder, val in vals.items():
                if val == outlier_value:
                    coder_counts[coder]["as_outlier"] += 1

    records = []
    total_possible = len(coder_data.unit_ids) * len(coder_data.variables)

    for coder in coder_data.coder_names:
        counts = coder_counts[coder]
        records.append({
            "coder": coder,
            "disagreements_involved": counts["involved"],
            "pct_involved": counts["involved"] / total_possible * 100 if total_possible > 0 else 0,
            "times_as_outlier": counts["as_outlier"],
            "outlier_rate": counts["as_outlier"] / counts["involved"] * 100 if counts["involved"] > 0 else 0,
        })

    return pd.DataFrame(records)
