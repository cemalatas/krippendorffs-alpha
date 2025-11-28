"""LLM client for Claude Opus 4.5 integration."""

import json
from typing import Any, Dict, List, Optional

from anthropic import Anthropic


# Model configuration
MODEL_ID = "claude-opus-4-5-20251101"
MAX_TOKENS = 4096


# System prompts for different roles
SYSTEM_PROMPTS = {
    "educational_companion": """You are an educational assistant helping researchers understand
intercoder reliability and Krippendorff's Alpha. Your explanations should:
1. Use clear, jargon-free language when possible
2. Explain "why it matters" for each concept
3. Reference Krippendorff's methodology when relevant
4. Provide actionable guidance for improvement

Education level: Medium (graduate student / early researcher)
Always explain implications, not just definitions.
Be concise but thorough.""",

    "data_detective": """You analyze CSV/data structures to identify:
- Which columns represent coder IDs
- Which columns represent unit IDs
- Which columns contain coded variables
- The measurement level of each variable (nominal, ordinal, interval, ratio)

Return structured JSON with your analysis and reasoning.
Be specific about column names and indices.""",

    "disagreement_analyst": """You analyze specific coding disagreements between coders.
For each disagreement, explain:
1. The nature of the disagreement
2. Possible reasons (ambiguous codebook? difficult case?)
3. Impact on reliability
4. Suggestions for resolution

Be constructive and educational in your analysis.""",

    "report_generator": """You generate academic-quality reports on intercoder reliability.
Ground all interpretations in Krippendorff's Content Analysis methodology.
Include:
- Executive summary
- Per-variable analysis with interpretations
- Pairwise coder comparisons
- Recommendations for improvement
- Properly cited interpretive thresholds (0.67 tentative, 0.80 acceptable)

Format using markdown with clear sections and tables where appropriate.""",

    "chat_assistant": """You are a helpful assistant answering questions about intercoder reliability,
Krippendorff's Alpha, and content analysis methodology.

You have expertise in:
- Krippendorff's Alpha calculation and interpretation
- Cohen's Kappa and Scott's Pi
- Measurement levels (nominal, ordinal, interval, ratio)
- Best practices for coder training and codebook development
- Interpreting reliability statistics

Keep answers concise but informative. Use examples when helpful.""",
}


class LLMClient:
    """Client for interacting with Claude Opus 4.5."""

    def __init__(self, api_key: str):
        """Initialize the LLM client.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key)
        self.model = MODEL_ID

    def call(
        self,
        prompt: str,
        system_prompt: str = "educational_companion",
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Make a simple text completion call.

        Args:
            prompt: User prompt
            system_prompt: Key from SYSTEM_PROMPTS or custom system prompt
            max_tokens: Maximum tokens in response

        Returns:
            Response text
        """
        system = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def call_json(
        self,
        prompt: str,
        system_prompt: str = "data_detective",
        max_tokens: int = MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Make a call expecting JSON response.

        Args:
            prompt: User prompt
            system_prompt: Key from SYSTEM_PROMPTS or custom system prompt
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON response
        """
        system = SYSTEM_PROMPTS.get(system_prompt, system_prompt)
        system += "\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown code blocks."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        return json.loads(text)

    def call_with_context(
        self,
        prompt: str,
        context: str,
        system_prompt: str = "educational_companion",
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Make a call with additional context.

        Args:
            prompt: User prompt
            context: Additional context (e.g., current results, data summary)
            system_prompt: Key from SYSTEM_PROMPTS or custom system prompt
            max_tokens: Maximum tokens in response

        Returns:
            Response text
        """
        system = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        full_prompt = f"""Context:
{context}

Question/Task:
{prompt}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": full_prompt}],
        )

        return response.content[0].text

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "chat_assistant",
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Multi-turn chat conversation.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            system_prompt: Key from SYSTEM_PROMPTS or custom system prompt
            max_tokens: Maximum tokens in response

        Returns:
            Response text
        """
        system = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

        return response.content[0].text


def get_chat_response(
    client: Optional[LLMClient],
    question: str,
    history: List[Dict[str, str]],
) -> str:
    """Get a chat response for the sidebar Q&A.

    Args:
        client: LLMClient instance (or None if not initialized)
        question: User's question
        history: Previous chat history

    Returns:
        Response text
    """
    if client is None:
        return "Please enter your API key on the Welcome page to enable chat."

    # Build messages from history + new question
    messages = history.copy()
    messages.append({"role": "user", "content": question})

    return client.chat(messages, system_prompt="chat_assistant")


def explain_alpha_value(client: LLMClient, alpha: float, variable: str, level: str) -> str:
    """Generate an educational explanation for an alpha value.

    Args:
        client: LLMClient instance
        alpha: The alpha value
        variable: Variable name
        level: Measurement level

    Returns:
        Explanation text
    """
    prompt = f"""Explain this Krippendorff's Alpha result:

Variable: {variable}
Measurement Level: {level}
Alpha Value: {alpha:.3f}

Provide:
1. Whether this is acceptable, tentative, or insufficient reliability
2. What this means practically for the research
3. Brief suggestions if improvement is needed"""

    return client.call(prompt, system_prompt="educational_companion")


def analyze_data_structure(client: LLMClient, df_preview: str, columns: List[str]) -> Dict[str, Any]:
    """Analyze uploaded data structure with LLM assistance.

    Args:
        client: LLMClient instance
        df_preview: String representation of DataFrame head
        columns: List of column names

    Returns:
        Analysis results as dict
    """
    prompt = f"""Analyze this data structure for intercoder reliability analysis:

Columns: {columns}

Data Preview:
{df_preview}

Identify:
1. Which column(s) likely contain unit IDs (article/case identifiers)
2. Which column(s) likely contain coder IDs
3. Which columns contain coded variables
4. For each coded variable, suggest a measurement level (nominal, ordinal, interval, ratio)

Return JSON with this structure:
{{
    "unit_id_columns": ["col1"],
    "coder_id_columns": ["col2"],
    "coded_variables": [
        {{"name": "var1", "suggested_level": "nominal", "reason": "..."}}
    ],
    "data_format": "wide" | "long" | "multi_file",
    "summary": "Brief description of the data structure"
}}"""

    return client.call_json(prompt, system_prompt="data_detective")


def analyze_disagreement(
    client: LLMClient,
    unit_id: str,
    variable: str,
    coder_values: Dict[str, Any],
) -> str:
    """Analyze a specific disagreement between coders.

    Args:
        client: LLMClient instance
        unit_id: Unit identifier
        variable: Variable name
        coder_values: Dict of coder -> value

    Returns:
        Analysis text
    """
    values_str = "\n".join([f"  - {coder}: {value}" for coder, value in coder_values.items()])

    prompt = f"""Analyze this coding disagreement:

Unit: {unit_id}
Variable: {variable}
Coder Values:
{values_str}

Explain:
1. The nature of this disagreement
2. Possible reasons for the disagreement
3. How this might affect reliability
4. Suggestions for resolving such disagreements in future coding"""

    return client.call(prompt, system_prompt="disagreement_analyst")


def generate_report(
    client: LLMClient,
    results_summary: str,
    study_metadata: Dict[str, str],
    theory_context: str = "",
) -> str:
    """Generate a comprehensive reliability report.

    Args:
        client: LLMClient instance
        results_summary: Summary of all computed statistics
        study_metadata: Dict with codebook_name, num_coders, procedure, etc.
        theory_context: Relevant excerpts from Krippendorff book

    Returns:
        Full report in markdown format
    """
    metadata_str = "\n".join([f"- {k}: {v}" for k, v in study_metadata.items()])

    prompt = f"""Generate a comprehensive intercoder reliability report.

Study Metadata:
{metadata_str}

Results Summary:
{results_summary}

{f"Theoretical Context (from Krippendorff):{chr(10)}{theory_context}" if theory_context else ""}

Generate a professional report with:
1. Executive Summary
2. Methodology Overview
3. Per-Variable Reliability Analysis
4. Pairwise Coder Agreement Analysis
5. Areas of Concern (low reliability variables)
6. Recommendations for Improvement
7. Conclusion

Use proper academic tone and cite standard thresholds (Krippendorff suggests 0.80 as acceptable,
0.67 as tentative minimum for drawing conclusions)."""

    return client.call(prompt, system_prompt="report_generator", max_tokens=8192)
