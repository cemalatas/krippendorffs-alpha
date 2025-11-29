"""Step 5: Disagreement Explorer."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Disagreements - K's Alpha Calculator", layout="wide")

st.title("Step 5: Explore Disagreements")

# Check prerequisites
if not st.session_state.app_state.get('analysis_complete'):
    st.warning("Please run the analysis on the Analysis page first.")
    st.stop()

coder_data = st.session_state.app_state['coder_data']
disagreements = st.session_state.app_state['disagreements']
disag_summary = st.session_state.app_state['disagreement_summary']

st.markdown(f"""
Understanding **where** coders disagree is just as important as knowing **how much** they disagree.
This page helps you identify specific cases of disagreement.

**Found {len(disagreements)} disagreements** across {len(coder_data.variables)} variables.
""")

with st.expander("Understanding Disagreement Types"):
    st.markdown("""
    | Type | Description |
    |------|-------------|
    | **Split** | Multiple distinct values, no clear majority |
    | **Outlier** | One coder differs from all others |
    | **Missing** | Disagreement involves missing values |
    """)

st.markdown("---")

# Summary heatmap
st.subheader("Disagreement Overview")

from core.disagreement_analyzer import get_disagreement_heatmap_data
from visualization.charts import plot_disagreement_heatmap

heatmap_df = get_disagreement_heatmap_data(coder_data)

fig = plot_disagreement_heatmap(heatmap_df)
st.plotly_chart(fig, use_container_width=True)

# Summary table
st.subheader("Disagreement Summary by Variable")

st.dataframe(
    disag_summary.style.background_gradient(subset=['pct_disagreements'], cmap='Reds'),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# Detailed explorer
st.subheader("Detailed Disagreement Explorer")

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    selected_var = st.selectbox(
        "Filter by Variable",
        options=["All"] + coder_data.variables,
    )

with col2:
    selected_type = st.selectbox(
        "Filter by Type",
        options=["All", "split", "outlier", "missing"],
    )

with col3:
    min_severity = st.slider(
        "Minimum Severity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )

# Apply filters
from core.disagreement_analyzer import filter_disagreements

filtered = filter_disagreements(
    disagreements,
    variable=selected_var if selected_var != "All" else None,
    disagreement_type=selected_type if selected_type != "All" else None,
    min_severity=min_severity,
)

st.markdown(f"**Showing {len(filtered)} of {len(disagreements)} disagreements**")

# Convert to DataFrame for display
if filtered:
    rows = []
    for d in filtered:
        row = {
            "Unit": d.unit_id,
            "Variable": d.variable,
            "Type": d.disagreement_type,
            "Severity": f"{d.severity:.2f}",
        }
        for coder in coder_data.coder_names:
            row[coder] = d.coder_values.get(coder, "-")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Highlight disagreements
    def highlight_disagreement(row):
        colors = []
        coder_cols = [c for c in row.index if c in coder_data.coder_names]

        # Get unique non-null values
        values = [row[c] for c in coder_cols if row[c] != "-" and row[c] is not None]
        unique_values = set(values)

        for col in row.index:
            if col in coder_cols:
                val = row[col]
                if val == "-" or val is None:
                    colors.append("background-color: #f0f0f0")
                elif len(unique_values) > 1:
                    # Find if this is the minority value
                    count = values.count(val)
                    if count == min([values.count(v) for v in unique_values]):
                        colors.append("background-color: #FFB6C1")  # Light red for outlier
                    else:
                        colors.append("background-color: #90EE90")  # Light green for majority
                else:
                    colors.append("")
            else:
                colors.append("")
        return colors

    styled_df = df.style.apply(highlight_disagreement, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # LLM Analysis of selected disagreement
    st.markdown("---")
    st.subheader("Analyze Specific Disagreement")

    selected_unit = st.selectbox(
        "Select a unit to analyze:",
        options=[d.unit_id for d in filtered[:50]],  # Limit to first 50
    )

    if selected_unit and st.session_state.app_state.get('llm_client'):
        selected_disag = next((d for d in filtered if d.unit_id == selected_unit), None)

        if selected_disag and st.button("Analyze This Disagreement"):
            with st.spinner("Analyzing..."):
                from core.llm_client import analyze_disagreement

                analysis = analyze_disagreement(
                    st.session_state.app_state['llm_client'],
                    selected_disag.unit_id,
                    selected_disag.variable,
                    selected_disag.coder_values,
                )

                st.markdown("### Analysis")
                st.markdown(analysis)

else:
    st.info("No disagreements match the current filters.")

# Export
st.markdown("---")
st.subheader("Export Disagreements")

if filtered:
    export_df = pd.DataFrame([d.to_dict() for d in filtered])

    csv = export_df.to_csv(index=False)
    st.download_button(
        "Download as CSV",
        csv,
        file_name="disagreements.csv",
        mime="text/csv",
    )
