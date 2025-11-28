"""Data models for Krippendorff's Alpha calculator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CoderData:
    """Canonical data structure for reliability analysis.

    Attributes:
        coder_names: List of coder identifiers (e.g., ["Duygu", "Elif", "Erin", "Nazli"])
        unit_ids: List of unit identifiers (e.g., article IDs, row numbers)
        variable_levels: Dict mapping variable names to measurement levels
        data: Dict mapping variable names to (n_coders, n_units) numpy arrays
    """
    coder_names: List[str]
    unit_ids: List[str]
    variable_levels: Dict[str, str]  # {"var_name": "nominal"|"ordinal"|"interval"|"ratio"}
    data: Dict[str, np.ndarray]  # {"var_name": 2D array (coders x units)}

    def get_reliability_matrix(self, variable: str) -> np.ndarray:
        """Return (n_coders, n_units) matrix for krippendorff format."""
        return self.data[variable]

    def get_pairwise_data(
        self, variable: str, coder_a: str, coder_b: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned arrays for two coders (for Cohen's Kappa)."""
        idx_a = self.coder_names.index(coder_a)
        idx_b = self.coder_names.index(coder_b)
        return self.data[variable][idx_a], self.data[variable][idx_b]

    @property
    def n_coders(self) -> int:
        return len(self.coder_names)

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def variables(self) -> List[str]:
        return list(self.variable_levels.keys())


@dataclass
class AlphaResult:
    """Result of Krippendorff's Alpha computation for a single variable."""
    variable: str
    level: str  # measurement level
    alpha_all_coders: float
    alpha_without: Dict[str, float] = field(default_factory=dict)  # {coder: alpha_if_removed}
    max_gain_if_removed: float = 0.0
    coder_whose_removal_helps_most: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "variable": self.variable,
            "level": self.level,
            "alpha_all_coders": self.alpha_all_coders,
            "max_gain_if_removed": self.max_gain_if_removed,
            "coder_whose_removal_helps_most": self.coder_whose_removal_helps_most,
        }
        for coder, alpha in self.alpha_without.items():
            result[f"alpha_without_{coder}"] = alpha
        return result


@dataclass
class PairwiseResult:
    """Result of pairwise Krippendorff's Alpha computation."""
    variable: str
    level: str
    coder_a: str
    coder_b: str
    alpha_pair: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "level": self.level,
            "coder_a": self.coder_a,
            "coder_b": self.coder_b,
            "alpha_pair": self.alpha_pair,
        }


@dataclass
class PairwiseOverall:
    """Summary statistics for a coder pair across all variables."""
    coder_a: str
    coder_b: str
    mean_alpha: float
    median_alpha: float
    min_alpha: float
    max_alpha: float
    std_alpha: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coder_a": self.coder_a,
            "coder_b": self.coder_b,
            "mean_alpha": self.mean_alpha,
            "median_alpha": self.median_alpha,
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
            "std_alpha": self.std_alpha,
        }


@dataclass
class Disagreement:
    """Details of a coding disagreement for a specific unit and variable."""
    unit_id: str
    variable: str
    coder_values: Dict[str, Any]  # {"Duygu": 1, "Elif": 2, ...}
    disagreement_type: str  # "split", "outlier", "missing"
    severity: float  # 0.0-1.0 based on value distance

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "unit_id": self.unit_id,
            "variable": self.variable,
            "disagreement_type": self.disagreement_type,
            "severity": self.severity,
        }
        result.update(self.coder_values)
        return result


@dataclass
class CoderImpact:
    """Analysis of a coder's impact on reliability."""
    coder: str
    mean_delta_when_removed: float
    median_delta_when_removed: float
    num_vars_removal_increases_alpha: int
    num_vars_removal_decreases_alpha: int
    largest_increase_from_removal: float
    largest_decrease_from_removal: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coder": self.coder,
            "mean_delta_when_removed": self.mean_delta_when_removed,
            "median_delta_when_removed": self.median_delta_when_removed,
            "num_vars_removal_increases_alpha": self.num_vars_removal_increases_alpha,
            "num_vars_removal_decreases_alpha": self.num_vars_removal_decreases_alpha,
            "largest_increase_from_removal": self.largest_increase_from_removal,
            "largest_decrease_from_removal": self.largest_decrease_from_removal,
        }


@dataclass
class AdditionalStats:
    """Additional reliability statistics beyond Krippendorff's Alpha."""
    variable: str
    percent_agreement_overall: float
    percent_agreement_pairwise: Dict[str, float]  # {"Coder_A-Coder_B": 0.85}
    cohens_kappa: Optional[Dict[str, float]] = None  # Pairwise only
    fleiss_kappa: Optional[float] = None  # Multi-coder

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "percent_agreement_overall": self.percent_agreement_overall,
            "percent_agreement_pairwise": self.percent_agreement_pairwise,
            "cohens_kappa": self.cohens_kappa,
            "fleiss_kappa": self.fleiss_kappa,
        }


def results_to_dataframe(results: List[AlphaResult]) -> pd.DataFrame:
    """Convert list of AlphaResult to DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


def pairwise_to_dataframe(results: List[PairwiseResult]) -> pd.DataFrame:
    """Convert list of PairwiseResult to DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


def disagreements_to_dataframe(disagreements: List[Disagreement]) -> pd.DataFrame:
    """Convert list of Disagreement to DataFrame."""
    return pd.DataFrame([d.to_dict() for d in disagreements])
