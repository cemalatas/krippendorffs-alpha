"""Step 1: Welcome & API Key Setup."""

import streamlit as st

st.set_page_config(page_title="Welcome - K's Alpha Calculator", layout="wide")

st.title("Welcome to the Intercoder Reliability Calculator")

st.markdown("""
This educational platform helps you calculate and understand **Krippendorff's Alpha**
and other intercoder reliability statistics for content analysis research.

## What is Intercoder Reliability?

Intercoder reliability (also called interrater reliability) measures how consistently
multiple coders classify the same content. It's essential for:

- Validating your coding scheme
- Ensuring your findings are reproducible
- Meeting publication standards

## What is Krippendorff's Alpha?

Krippendorff's Alpha (α) is a reliability coefficient developed for content analysis. It:

- **Accounts for chance agreement** (unlike simple percent agreement)
- **Works with any number of coders** (2 or more)
- **Handles missing data** gracefully
- **Supports different measurement levels**: nominal, ordinal, interval, and ratio
""")

with st.expander("Learn More: Interpreting Alpha Values"):
    st.markdown("""
    | Alpha Value | Interpretation | Recommendation |
    |-------------|----------------|----------------|
    | α ≥ 0.80 | **Acceptable** reliability | Proceed with analysis |
    | 0.67 ≤ α < 0.80 | **Tentative** reliability | Report with caution |
    | α < 0.67 | **Insufficient** reliability | Revise codebook or retrain coders |

    *Thresholds based on Krippendorff (2019). Content Analysis: An Introduction to Its Methodology.*
    """)

st.markdown("---")

# API Key Setup
st.header("Setup: API Key (Optional)")

st.markdown("""
This app can use **Claude Opus 4.5** to provide educational explanations and generate AI-powered reports.
Your API key is stored only in your browser session and never saved.

**Note:** The API key is optional. You can skip this step and still use all calculation features.
AI-powered explanations will be disabled if you skip.
""")

api_key = st.text_input(
    "Anthropic API Key",
    type="password",
    value=st.session_state.app_state.get('api_key', '') or '',
    help="Get your API key from https://console.anthropic.com/",
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Save API Key", type="primary"):
        if api_key.strip():
            try:
                from core.llm_client import LLMClient

                # Test the API key
                client = LLMClient(api_key.strip())
                test_response = client.call("Say 'API key validated' in exactly 3 words.")

                st.session_state.app_state['api_key'] = api_key.strip()
                st.session_state.app_state['llm_client'] = client
                st.session_state.app_state['current_step'] = 2

                st.success("API key validated! You can now proceed to upload your data.")
                st.info("Navigate to **2. Data Upload** in the sidebar to continue.")

            except Exception as e:
                st.error(f"Failed to validate API key: {str(e)}")
        else:
            st.warning("Please enter a valid API key.")

with col2:
    if st.button("Skip (Continue without AI)", type="secondary"):
        st.session_state.app_state['api_key'] = None
        st.session_state.app_state['llm_client'] = None
        st.session_state.app_state['current_step'] = 2
        st.success("Continuing without AI features. All calculation features are available.")
        st.info("Navigate to **2. Data Upload** in the sidebar to continue.")

# Show current status
st.markdown("---")
st.subheader("Current Status")

if st.session_state.app_state.get('api_key'):
    st.success("✅ API key configured - AI features enabled")
elif st.session_state.app_state.get('current_step', 1) >= 2:
    st.info("ℹ️ AI features skipped - calculations still available")
else:
    st.info("⬜ API key not yet configured (optional)")

if st.session_state.app_state.get('coder_data'):
    st.success(f"✅ Data loaded: {st.session_state.app_state['coder_data'].n_coders} coders, {st.session_state.app_state['coder_data'].n_units} units")
else:
    st.info("⬜ No data loaded yet")

if st.session_state.app_state.get('analysis_complete'):
    st.success("✅ Analysis complete")
else:
    st.info("⬜ Analysis not yet run")
