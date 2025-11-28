"""Step 4: Run Analysis."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Run Analysis - K's Alpha Calculator", layout="wide")

st.title("Step 4: Run Reliability Analysis")

# Check prerequisites
if not st.session_state.app_state.get('coder_data'):
    st.warning("Please configure your variables on the Configure Variables page first.")
    st.stop()

coder_data = st.session_state.app_state['coder_data']

st.markdown(f"""
Ready to compute reliability statistics for your data:

- **{coder_data.n_coders} coders**: {', '.join(coder_data.coder_names)}
- **{coder_data.n_units} units**
- **{len(coder_data.variables)} variables**
""")

# Data verification section
with st.expander("🔍 Verify Data Before Analysis", expanded=False):
    st.markdown("Use this to verify that your data was loaded correctly.")

    verify_var = st.selectbox(
        "Select a variable to inspect:",
        options=coder_data.variables,
        key="verify_var"
    )

    if verify_var:
        matrix = coder_data.get_reliability_matrix(verify_var)
        level = coder_data.variable_levels[verify_var]

        st.markdown(f"**Variable:** {verify_var}")
        st.markdown(f"**Measurement Level:** {level}")
        st.markdown(f"**Matrix Shape:** {matrix.shape[0]} coders × {matrix.shape[1]} units")

        # Show unique values
        flat = matrix[np.isfinite(matrix)]
        unique_vals = sorted(pd.unique(flat))
        st.markdown(f"**Unique Values:** {unique_vals}")

        # Show matrix preview
        st.markdown("**Data Matrix (first 10 units):**")
        preview_df = pd.DataFrame(
            matrix[:, :min(10, matrix.shape[1])],
            index=coder_data.coder_names,
            columns=[f"Unit {i+1}" for i in range(min(10, matrix.shape[1]))]
        )
        st.dataframe(preview_df)

        # Quick single-variable alpha calculation
        import krippendorff as kripp
        try:
            single_alpha = kripp.alpha(reliability_data=matrix, level_of_measurement=level)
            st.metric(f"K's Alpha for {verify_var}", f"{single_alpha:.6f}")
            st.caption("Compare this value with ReCal to verify correct data loading.")
        except Exception as e:
            st.error(f"Could not compute: {e}")

with st.expander("What will be calculated?"):
    st.markdown("""
    **Primary Metrics:**
    - **Overall Krippendorff's Alpha** - Aggregate reliability across all variables
    - **Per-variable Krippendorff's Alpha** - Reliability for each coded variable
    - **Pairwise Alpha** - Agreement between each pair of coders

    **Additional Statistics:**
    - **Percent Agreement** - Simple agreement rate (baseline)
    - **Cohen's Kappa** - Chance-corrected pairwise agreement
    - **Fleiss' Kappa** - Multi-coder agreement (generalization of Scott's Pi)

    **Coder Analysis:**
    - **Coder Impact** - How each coder affects overall reliability
    - **Disagreement Detection** - Per-row identification of disagreements
    """)

st.markdown("---")

# Run Analysis Button
if st.button("Run Complete Analysis", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Per-variable alpha
        status_text.text("Computing per-variable Krippendorff's Alpha...")
        from core.krippendorff import (
            compute_per_variable_alpha,
            compute_pairwise_alpha,
            compute_pairwise_overall,
            compute_coder_impact,
            compute_overall_alpha,
        )

        per_var_results = compute_per_variable_alpha(coder_data)
        st.session_state.app_state['per_variable_results'] = per_var_results
        progress_bar.progress(20)

        # Step 2: Overall alpha
        status_text.text("Computing overall Krippendorff's Alpha...")
        overall_alpha = compute_overall_alpha(per_var_results)
        st.session_state.app_state['overall_alpha'] = overall_alpha
        progress_bar.progress(35)

        # Step 3: Pairwise alpha
        status_text.text("Computing pairwise reliability...")
        pairwise_results = compute_pairwise_alpha(coder_data)
        st.session_state.app_state['pairwise_results'] = pairwise_results

        pairwise_overall = compute_pairwise_overall(pairwise_results)
        st.session_state.app_state['pairwise_overall'] = pairwise_overall
        progress_bar.progress(50)

        # Step 4: Coder impact
        status_text.text("Analyzing coder impact...")
        coder_impact_results = compute_coder_impact(per_var_results, coder_data.coder_names)
        st.session_state.app_state['coder_impact'] = coder_impact_results
        progress_bar.progress(65)

        # Step 5: Additional statistics
        status_text.text("Computing additional statistics (Kappa, Pi, Agreement)...")
        from core.additional_stats import compute_all_additional_stats

        additional_stats = {}
        for var in coder_data.variables:
            additional_stats[var] = compute_all_additional_stats(coder_data, var)
        st.session_state.app_state['additional_stats'] = additional_stats
        progress_bar.progress(80)

        # Step 6: Find disagreements
        status_text.text("Identifying disagreements...")
        from core.disagreement_analyzer import find_all_disagreements, disagreement_summary

        disagreements = find_all_disagreements(coder_data)
        st.session_state.app_state['disagreements'] = disagreements

        disag_summary = disagreement_summary(coder_data)
        st.session_state.app_state['disagreement_summary'] = disag_summary
        progress_bar.progress(100)

        st.session_state.app_state['analysis_complete'] = True
        st.session_state.app_state['current_step'] = max(st.session_state.app_state.get('current_step', 1), 4)

        status_text.text("Analysis complete!")
        st.success("All reliability statistics computed successfully!")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Show results summary if analysis is complete
if st.session_state.app_state.get('analysis_complete'):
    st.markdown("---")
    st.header("Quick Results Summary")

    overall_alpha = st.session_state.app_state['overall_alpha']
    per_var = st.session_state.app_state['per_variable_results']
    disagreements = st.session_state.app_state['disagreements']

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        from core.krippendorff import interpret_alpha
        interp, color = interpret_alpha(overall_alpha)
        st.metric(
            "Overall K's Alpha",
            f"{overall_alpha:.3f}",
            help="Mean of all per-variable alphas. For single-variable comparison with ReCal, check Per-Variable Results below.",
        )
        st.markdown(f":{color}[{interp}]")

    with col2:
        acceptable = sum(1 for r in per_var if r.alpha_all_coders >= 0.80)
        st.metric(
            "Acceptable Variables",
            f"{acceptable}/{len(per_var)}",
            help="Variables with α ≥ 0.80",
        )

    with col3:
        low_alpha = sum(1 for r in per_var if r.alpha_all_coders < 0.67)
        st.metric(
            "Problem Variables",
            f"{low_alpha}",
            help="Variables with α < 0.67",
        )

    with col4:
        st.metric(
            "Total Disagreements",
            len(disagreements),
            help="Units with coder disagreement",
        )

    # Quick interpretation
    if st.session_state.app_state.get('llm_client'):
        with st.expander("AI Interpretation", expanded=True):
            if st.button("Generate Interpretation"):
                with st.spinner("Generating interpretation..."):
                    from core.krippendorff import results_summary

                    summary = results_summary(
                        per_var,
                        st.session_state.app_state['pairwise_overall'],
                        st.session_state.app_state['coder_impact'],
                    )

                    response = st.session_state.app_state['llm_client'].call_with_context(
                        "Provide a brief interpretation of these reliability results. "
                        "Highlight key findings, areas of concern, and recommendations.",
                        summary,
                        system_prompt="educational_companion",
                    )

                    st.markdown(response)

    # Per-variable preview
    st.subheader("Per-Variable Results Preview")
    st.caption("Compare individual Alpha values with ReCal for verification (use 6-decimal values).")

    df = pd.DataFrame([{
        "Variable": r.variable,
        "Alpha": r.alpha_all_coders,
        "Alpha (6 decimals)": f"{r.alpha_all_coders:.6f}",
        "Level": r.level,
        "Status": interpret_alpha(r.alpha_all_coders)[0],
    } for r in per_var])

    st.dataframe(
        df.style.format({"Alpha": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.info("Navigate to **5. Disagreements** to explore per-row disagreements, or **6. Results** for full dashboard.")
