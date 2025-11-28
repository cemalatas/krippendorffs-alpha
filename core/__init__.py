"""Core computation modules for Krippendorff's Alpha calculator."""

from .models import CoderData, AlphaResult, Disagreement, PairwiseResult
from .llm_client import LLMClient, get_chat_response
from .data_transformer import transform_to_coder_data, detect_format
from .krippendorff import (
    compute_alpha,
    compute_per_variable_alpha,
    compute_pairwise_alpha,
    compute_pairwise_overall,
    compute_coder_impact,
    compute_overall_alpha,
)
from .additional_stats import (
    compute_percent_agreement,
    compute_cohens_kappa,
    compute_fleiss_kappa,
)
from .disagreement_analyzer import find_disagreements, disagreement_summary

__all__ = [
    'CoderData',
    'AlphaResult',
    'Disagreement',
    'PairwiseResult',
    'LLMClient',
    'get_chat_response',
    'transform_to_coder_data',
    'detect_format',
    'compute_alpha',
    'compute_per_variable_alpha',
    'compute_pairwise_alpha',
    'compute_pairwise_overall',
    'compute_coder_impact',
    'compute_overall_alpha',
    'compute_percent_agreement',
    'compute_cohens_kappa',
    'compute_fleiss_kappa',
    'find_disagreements',
    'disagreement_summary',
]
