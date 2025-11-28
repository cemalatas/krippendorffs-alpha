"""Step 6: Results Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Results - K's Alpha Calculator", layout="wide")

st.title("Step 6: Results Dashboard")

# Check prerequisites
if not st.session_state.app_state.get('analysis_complete'):
    st.warning("Please run the analysis on the Analysis page first.")
    st.stop()

coder_data = st.session_state.app_state['coder_data']
per_var = st.session_state.app_state['per_variable_results']
pairwise_overall = st.session_state.app_state['pairwise_overall']
coder_impact = st.session_state.app_state['coder_impact']
additional_stats = st.session_state.app_state['additional_stats']
overall_alpha = st.session_state.app_state['overall_alpha']

from core.krippendorff import interpret_alpha
from visualization.charts import (
    plot_per_variable_alpha,
    plot_pairwise_heatmap,
    plot_coder_impact,
    plot_confusion_matrix,
    color_by_alpha,
)
from core.additional_stats import compute_confusion_matrix

# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Per-Variable",
    "Pairwise Analysis",
    "Coder Performance",
    "Additional Statistics",
])

with tab1:
    st.header("Reliability Overview")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        interp, color = interpret_alpha(overall_alpha)
        st.metric("Overall K's Alpha", f"{overall_alpha:.3f}",
                  help="Mean alpha across all variables")
        st.markdown(f"**Status:** :{color}[{interp}]")

    with col2:
        # Mean percent agreement
        mean_agree = np.mean([s.percent_agreement_overall for s in additional_stats.values()])
        st.metric("Mean % Agreement", f"{mean_agree:.1%}")

    with col3:
        acceptable = sum(1 for r in per_var if r.alpha_all_coders >= 0.80)
        total = len(per_var)
        st.metric("Acceptable Variables", f"{acceptable}/{total}")

    with col4:
        problems = sum(1 for r in per_var if r.alpha_all_coders < 0.67)
        st.metric("Problem Variables", f"{problems}")

    # LLM interpretation
    if st.session_state.app_state.get('llm_client'):
        with st.expander("AI Summary", expanded=True):
            if 'overview_interpretation' not in st.session_state:
                from core.krippendorff import results_summary
                summary = results_summary(per_var, pairwise_overall, coder_impact)

                response = st.session_state.app_state['llm_client'].call_with_context(
                    "Provide a 2-3 paragraph executive summary of these reliability results. "
                    "Focus on overall quality, key concerns, and actionable insights.",
                    summary,
                )
                st.session_state['overview_interpretation'] = response

            st.markdown(st.session_state['overview_interpretation'])

    # Distribution chart
    st.subheader("Alpha Value Distribution")
    fig = plot_per_variable_alpha(per_var)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Per-Variable Reliability")

    # Create detailed DataFrame
    df = pd.DataFrame([{
        "Variable": r.variable,
        "Level": r.level,
        "K's Alpha": r.alpha_all_coders,
        "Status": interpret_alpha(r.alpha_all_coders)[0],
        "Max Gain if Removed": r.max_gain_if_removed,
        "Coder to Remove": r.coder_whose_removal_helps_most,
    } for r in per_var])

    # Apply styling
    styled = df.style.map(
        lambda x: color_by_alpha(x) if isinstance(x, (int, float)) and not pd.isna(x) else "",
        subset=["K's Alpha"]
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Variable detail
    st.subheader("Variable Detail")

    selected_var = st.selectbox("Select variable for details:", coder_data.variables)

    if selected_var:
        var_result = next((r for r in per_var if r.variable == selected_var), None)
        var_stats = additional_stats.get(selected_var)

        if var_result:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("K's Alpha", f"{var_result.alpha_all_coders:.3f}")

            with col2:
                if var_stats and var_stats.fleiss_kappa:
                    st.metric("Fleiss' Kappa", f"{var_stats.fleiss_kappa:.3f}")

            with col3:
                if var_stats:
                    st.metric("% Agreement", f"{var_stats.percent_agreement_overall:.1%}")

            # Alpha without each coder
            st.markdown("**Alpha if coder removed:**")
            cols = st.columns(len(coder_data.coder_names))
            for i, coder in enumerate(coder_data.coder_names):
                with cols[i]:
                    alpha_without = var_result.alpha_without.get(coder, np.nan)
                    delta = alpha_without - var_result.alpha_all_coders
                    st.metric(
                        coder,
                        f"{alpha_without:.3f}",
                        delta=f"{delta:+.3f}",
                        delta_color="normal" if delta < 0 else "inverse",
                    )

with tab3:
    st.header("Pairwise Coder Analysis")

    # Pairwise heatmap
    fig = plot_pairwise_heatmap(pairwise_overall)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed pairwise table
    st.subheader("Pairwise Statistics")

    pairwise_df = pd.DataFrame([{
        "Coder A": p.coder_a,
        "Coder B": p.coder_b,
        "Mean Alpha": f"{p.mean_alpha:.3f}",
        "Median Alpha": f"{p.median_alpha:.3f}",
        "Min Alpha": f"{p.min_alpha:.3f}",
        "Max Alpha": f"{p.max_alpha:.3f}",
    } for p in pairwise_overall])

    st.dataframe(pairwise_df, use_container_width=True, hide_index=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")

    col1, col2, col3 = st.columns(3)

    with col1:
        cm_var = st.selectbox("Variable:", coder_data.variables, key="cm_var")
    with col2:
        cm_coder_a = st.selectbox("Coder A:", coder_data.coder_names, key="cm_a")
    with col3:
        available_coders_b = [c for c in coder_data.coder_names if c != cm_coder_a]
        cm_coder_b = st.selectbox("Coder B:", available_coders_b, key="cm_b")

    if cm_var and cm_coder_a and cm_coder_b:
        conf_matrix, categories = compute_confusion_matrix(
            coder_data, cm_var, cm_coder_a, cm_coder_b
        )
        fig = plot_confusion_matrix(conf_matrix, categories, cm_coder_a, cm_coder_b, cm_var)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Coder Performance")

    st.markdown("""
    This analysis shows how each coder impacts overall reliability.
    **Positive delta** = removing this coder improves reliability (they may need additional training).
    **Negative delta** = removing this coder hurts reliability (they are reliable).
    """)

    # Impact chart
    fig = plot_coder_impact(coder_impact)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Coder Impact Details")

    coder_df = pd.DataFrame([{
        "Coder": c.coder,
        "Mean Delta": f"{c.mean_delta_when_removed:+.3f}",
        "Variables Improved": c.num_vars_removal_increases_alpha,
        "Variables Hurt": c.num_vars_removal_decreases_alpha,
        "Largest Improvement": f"{c.largest_increase_from_removal:+.3f}",
        "Largest Decrease": f"{c.largest_decrease_from_removal:+.3f}",
    } for c in coder_impact])

    st.dataframe(coder_df, use_container_width=True, hide_index=True)

    # LLM interpretation
    if st.session_state.app_state.get('llm_client'):
        with st.expander("AI Analysis of Coder Performance"):
            if st.button("Analyze Coder Performance"):
                with st.spinner("Analyzing..."):
                    coder_summary = "\n".join([
                        f"- {c.coder}: mean delta {c.mean_delta_when_removed:+.3f}, "
                        f"improves {c.num_vars_removal_increases_alpha} vars when removed"
                        for c in coder_impact
                    ])

                    response = st.session_state.app_state['llm_client'].call(
                        f"Analyze this coder impact data and provide specific recommendations:\n\n{coder_summary}",
                        system_prompt="educational_companion",
                    )

                    st.markdown(response)

with tab5:
    st.header("Additional Statistics")

    st.markdown("""
    Beyond Krippendorff's Alpha, these statistics provide additional perspectives on reliability.
    """)

    with st.expander("Understanding These Metrics"):
        st.markdown("""
        | Metric | Description | When to Use |
        |--------|-------------|-------------|
        | **% Agreement** | Simple proportion of matching codes | Quick baseline |
        | **Cohen's Kappa** | Chance-corrected pairwise agreement | 2 coders, nominal data |
        | **Fleiss' Kappa** | Multi-coder generalization | 3+ coders |
        | **K's Alpha** | Most robust, handles missing data | General use |
        """)

    # Comparison table
    comparison_data = []
    for var in coder_data.variables:
        var_result = next((r for r in per_var if r.variable == var), None)
        var_stats = additional_stats.get(var)

        row = {
            "Variable": var,
            "K's Alpha": var_result.alpha_all_coders if var_result else np.nan,
            "% Agreement": var_stats.percent_agreement_overall if var_stats else np.nan,
            "Fleiss' Kappa": var_stats.fleiss_kappa if var_stats else np.nan,
        }

        # Add mean Cohen's Kappa
        if var_stats and var_stats.cohens_kappa:
            kappas = [v for v in var_stats.cohens_kappa.values() if not np.isnan(v)]
            row["Mean Cohen's κ"] = np.mean(kappas) if kappas else np.nan
        else:
            row["Mean Cohen's κ"] = np.nan

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(
        comparison_df.style.format({
            "K's Alpha": "{:.3f}",
            "% Agreement": "{:.1%}",
            "Fleiss' Kappa": "{:.3f}",
            "Mean Cohen's κ": "{:.3f}",
        }).background_gradient(subset=["K's Alpha"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

# Export section
st.markdown("---")
st.subheader("Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Per-variable CSV
    per_var_df = pd.DataFrame([r.to_dict() for r in per_var])
    csv1 = per_var_df.to_csv(index=False)
    st.download_button(
        "Download Per-Variable Results (CSV)",
        csv1,
        file_name="per_variable_alpha.csv",
        mime="text/csv",
    )

with col2:
    # Pairwise CSV
    pairwise_df_export = pd.DataFrame([p.to_dict() for p in pairwise_overall])
    csv2 = pairwise_df_export.to_csv(index=False)
    st.download_button(
        "Download Pairwise Results (CSV)",
        csv2,
        file_name="pairwise_alpha.csv",
        mime="text/csv",
    )

with col3:
    # Coder impact CSV
    coder_df_export = pd.DataFrame([c.to_dict() for c in coder_impact])
    csv3 = coder_df_export.to_csv(index=False)
    st.download_button(
        "Download Coder Impact (CSV)",
        csv3,
        file_name="coder_impact.csv",
        mime="text/csv",
    )
