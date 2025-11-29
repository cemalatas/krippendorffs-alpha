"""Plotly visualization functions for reliability analysis."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from core.models import AlphaResult, CoderImpact, PairwiseOverall


def truncate_label(label: str, max_length: int = 20) -> str:
    """Truncate a label if it exceeds max_length."""
    if len(label) > max_length:
        return label[:max_length - 3] + "..."
    return label


def plot_per_variable_alpha(
    results: List[AlphaResult],
    title: str = "Krippendorff's Alpha by Variable",
    max_label_length: int = 25,
) -> go.Figure:
    """Create horizontal bar chart of alpha values per variable.

    Args:
        results: List of AlphaResult objects
        title: Chart title
        max_label_length: Maximum length for variable labels before truncation

    Returns:
        Plotly Figure
    """
    df = pd.DataFrame([{
        "variable": r.variable,
        "variable_truncated": truncate_label(r.variable, max_label_length),
        "alpha": r.alpha_all_coders,
        "level": r.level,
    } for r in results])

    # Sort by alpha value
    df = df.sort_values("alpha", ascending=True)

    # Color mapping for measurement levels
    color_map = {
        "nominal": "#636EFA",
        "ordinal": "#EF553B",
        "interval": "#00CC96",
        "ratio": "#AB63FA",
    }

    fig = px.bar(
        df,
        x="alpha",
        y="variable_truncated",
        color="level",
        orientation="h",
        color_discrete_map=color_map,
        title=title,
        labels={"alpha": "Krippendorff's Alpha", "variable_truncated": "Variable", "level": "Level"},
        hover_data={"variable": True, "variable_truncated": False},
    )

    # Add threshold lines
    fig.add_vline(
        x=0.67,
        line_dash="dash",
        line_color="orange",
        annotation_text="Tentative (0.67)",
        annotation_position="top",
    )
    fig.add_vline(
        x=0.80,
        line_dash="solid",
        line_color="green",
        annotation_text="Acceptable (0.80)",
        annotation_position="top",
    )

    fig.update_layout(
        xaxis_range=[0, 1.05],
        height=max(400, len(df) * 25),
        showlegend=True,
        legend_title_text="Measurement Level",
    )

    return fig


def plot_pairwise_heatmap(
    pairwise_overall: List[PairwiseOverall],
    metric: str = "mean_alpha",
    title: str = "Pairwise Coder Agreement",
) -> go.Figure:
    """Create symmetric heatmap of pairwise agreement.

    Args:
        pairwise_overall: List of PairwiseOverall objects
        metric: Which metric to display ("mean_alpha", "median_alpha", etc.)
        title: Chart title

    Returns:
        Plotly Figure
    """
    # Get unique coders
    coders = sorted(set(
        [p.coder_a for p in pairwise_overall] +
        [p.coder_b for p in pairwise_overall]
    ))
    n = len(coders)

    # Build symmetric matrix
    matrix = np.eye(n)  # Diagonal = 1.0

    for p in pairwise_overall:
        i = coders.index(p.coder_a)
        j = coders.index(p.coder_b)
        value = getattr(p, metric)
        matrix[i, j] = matrix[j, i] = value

    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=coders,
        y=coders,
        colorscale="RdYlGn",
        showscale=True,
        annotation_text=np.round(matrix, 2).astype(str),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Coder",
        yaxis_title="Coder",
        height=400,
    )

    return fig


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    categories: List,
    coder_a: str,
    coder_b: str,
    variable: str,
) -> go.Figure:
    """Create annotated confusion matrix heatmap.

    Args:
        conf_matrix: 2D confusion matrix array
        categories: List of category labels
        coder_a: First coder name (y-axis)
        coder_b: Second coder name (x-axis)
        variable: Variable name

    Returns:
        Plotly Figure
    """
    # Convert categories to strings
    cat_labels = [str(c) for c in categories]

    fig = ff.create_annotated_heatmap(
        z=conf_matrix,
        x=cat_labels,
        y=cat_labels,
        colorscale="Blues",
        showscale=True,
    )

    fig.update_layout(
        title=f"Confusion Matrix: {coder_a} vs {coder_b} ({variable})",
        xaxis_title=coder_b,
        yaxis_title=coder_a,
        height=400,
    )

    return fig


def plot_disagreement_heatmap(
    heatmap_df: pd.DataFrame,
    title: str = "Disagreement Count by Variable and Coder Pair",
    max_label_length: int = 25,
) -> go.Figure:
    """Create heatmap of disagreement counts.

    Args:
        heatmap_df: DataFrame with variables as index, coder pairs as columns
        title: Chart title
        max_label_length: Maximum length for variable labels before truncation

    Returns:
        Plotly Figure
    """
    # Truncate long variable names
    truncated_vars = [truncate_label(str(v), max_label_length) for v in heatmap_df.index]
    # Keep full names for hover
    full_vars = heatmap_df.index.tolist()

    fig = px.imshow(
        heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=truncated_vars,
        color_continuous_scale="Reds",
        title=title,
        labels={"x": "Coder Pair", "y": "Variable", "color": "Disagreements"},
    )

    # Add full variable names to hover
    fig.update_traces(
        hovertemplate="Variable: %{customdata}<br>Coder Pair: %{x}<br>Disagreements: %{z}<extra></extra>",
        customdata=[[v] * len(heatmap_df.columns) for v in full_vars],
    )

    fig.update_layout(
        height=max(400, len(heatmap_df) * 25),
    )

    return fig


def plot_coder_impact(
    coder_impact: List[CoderImpact],
    title: str = "Coder Impact on Reliability",
) -> go.Figure:
    """Create bar chart showing coder impact.

    Positive delta = removing coder improves reliability (problematic coder).
    Negative delta = removing coder hurts reliability (reliable coder).

    Args:
        coder_impact: List of CoderImpact objects
        title: Chart title

    Returns:
        Plotly Figure
    """
    df = pd.DataFrame([{
        "coder": c.coder,
        "delta": c.mean_delta_when_removed,
        "vars_improved": c.num_vars_removal_increases_alpha,
        "vars_hurt": c.num_vars_removal_decreases_alpha,
    } for c in coder_impact])

    # Color based on impact
    df["color"] = df["delta"].apply(
        lambda x: "red" if x > 0.02 else ("green" if x < -0.02 else "gray")
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["coder"],
        y=df["delta"],
        marker_color=df["color"],
        text=df["delta"].apply(lambda x: f"{x:+.3f}"),
        textposition="outside",
        hovertemplate=(
            "Coder: %{x}<br>"
            "Mean Delta: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Coder",
        yaxis_title="Mean Delta (alpha when removed - alpha with all)",
        height=400,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text="↑ Positive = Removing improves reliability",
                showarrow=False,
                font=dict(size=10, color="red"),
            ),
            dict(
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                text="↓ Negative = Removing hurts reliability",
                showarrow=False,
                font=dict(size=10, color="green"),
            ),
        ],
    )

    return fig


def plot_alpha_distribution(
    results: List[AlphaResult],
    title: str = "Distribution of Alpha Values",
) -> go.Figure:
    """Create histogram of alpha values across variables.

    Args:
        results: List of AlphaResult objects
        title: Chart title

    Returns:
        Plotly Figure
    """
    alphas = [r.alpha_all_coders for r in results if not np.isnan(r.alpha_all_coders)]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=alphas,
        nbinsx=20,
        marker_color="#636EFA",
        name="Variables",
    ))

    # Add threshold lines
    fig.add_vline(x=0.67, line_dash="dash", line_color="orange", annotation_text="0.67")
    fig.add_vline(x=0.80, line_dash="solid", line_color="green", annotation_text="0.80")

    fig.update_layout(
        title=title,
        xaxis_title="Krippendorff's Alpha",
        yaxis_title="Number of Variables",
        xaxis_range=[0, 1.05],
        height=350,
    )

    return fig


def plot_agreement_comparison(
    variables: List[str],
    alpha_values: List[float],
    kappa_values: List[float],
    agreement_values: List[float],
    title: str = "Comparison of Reliability Metrics",
) -> go.Figure:
    """Create grouped bar chart comparing different reliability metrics.

    Args:
        variables: List of variable names
        alpha_values: Krippendorff's Alpha values
        kappa_values: Fleiss' Kappa values
        agreement_values: Percent agreement values
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="K's Alpha",
        x=variables,
        y=alpha_values,
        marker_color="#636EFA",
    ))

    fig.add_trace(go.Bar(
        name="Fleiss' Kappa",
        x=variables,
        y=kappa_values,
        marker_color="#EF553B",
    ))

    fig.add_trace(go.Bar(
        name="% Agreement",
        x=variables,
        y=agreement_values,
        marker_color="#00CC96",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Variable",
        yaxis_title="Value",
        barmode="group",
        height=400,
        yaxis_range=[0, 1.05],
    )

    return fig


def color_by_alpha(val: float) -> str:
    """Return CSS color string based on alpha value.

    Args:
        val: Alpha value

    Returns:
        CSS color string
    """
    if pd.isna(val):
        return "background-color: #f0f0f0"
    elif val >= 0.80:
        return "background-color: #90EE90"  # Light green
    elif val >= 0.67:
        return "background-color: #FFE4B5"  # Light orange
    else:
        return "background-color: #FFB6C1"  # Light red
