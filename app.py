"""
Krippendorff's Alpha Educational Calculator
A Streamlit app for calculating intercoder reliability with LLM-guided education.
"""

import streamlit as st

st.set_page_config(
    page_title="Krippendorff's Alpha Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        # Auth
        'api_key': None,
        'llm_client': None,

        # Data
        'raw_dataframes': {},
        'coder_data': None,
        'column_mapping': {},
        'measurement_levels': {},
        'variable_levels': [],

        # Results
        'overall_alpha': None,
        'per_variable_results': None,
        'pairwise_results': None,
        'pairwise_overall': None,
        'coder_impact': None,
        'additional_stats': {},
        'disagreements': [],

        # Report
        'generated_report': None,

        # Chat
        'chat_history': [],

        # Navigation
        'current_step': 1,
        'analysis_complete': False,
    }

if 'chat_input_key' not in st.session_state:
    st.session_state.chat_input_key = 0


def render_sidebar_chat():
    """Render the sidebar chat for Q&A with Claude."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ask Claude")

    if st.session_state.app_state['api_key'] is None:
        st.sidebar.info("Enter your API key on the Welcome page to enable chat.")
        return

    # Chat input
    user_question = st.sidebar.text_input(
        "Ask a question about reliability analysis:",
        key=f"chat_input_{st.session_state.chat_input_key}",
        placeholder="e.g., What is a good alpha value?"
    )

    if st.sidebar.button("Ask", key="ask_button"):
        if user_question.strip():
            from core.llm_client import get_chat_response

            with st.sidebar.spinner("Thinking..."):
                response = get_chat_response(
                    st.session_state.app_state['llm_client'],
                    user_question,
                    st.session_state.app_state['chat_history']
                )

            # Add to history
            st.session_state.app_state['chat_history'].append({
                'role': 'user',
                'content': user_question
            })
            st.session_state.app_state['chat_history'].append({
                'role': 'assistant',
                'content': response
            })

            # Increment key to clear input
            st.session_state.chat_input_key += 1
            st.rerun()

    # Display chat history (most recent first, limited to last 6)
    if st.session_state.app_state['chat_history']:
        st.sidebar.markdown("**Recent Q&A:**")
        history = st.session_state.app_state['chat_history'][-6:]

        for i in range(len(history) - 1, -1, -2):
            if i >= 1:
                with st.sidebar.expander(f"Q: {history[i-1]['content'][:50]}...", expanded=False):
                    st.markdown(f"**Q:** {history[i-1]['content']}")
                    st.markdown(f"**A:** {history[i]['content']}")


def render_progress_indicator():
    """Render the step progress indicator in sidebar."""
    st.sidebar.markdown("## Progress")

    steps = [
        ("1. Welcome", 1),
        ("2. Data Upload", 2),
        ("3. Configure Variables", 3),
        ("4. Run Analysis", 4),
        ("5. Disagreements", 5),
        ("6. Results", 6),
        ("7. Report", 7),
    ]

    current = st.session_state.app_state['current_step']

    for name, step_num in steps:
        if step_num < current:
            st.sidebar.markdown(f"✅ {name}")
        elif step_num == current:
            st.sidebar.markdown(f"**➡️ {name}**")
        else:
            st.sidebar.markdown(f"⬜ {name}")


# Main page content
st.title("📊 Krippendorff's Alpha Calculator")
st.markdown("""
Welcome to the **Intercoder Reliability Educational Platform**!

This tool helps you:
- **Calculate** Krippendorff's Alpha and other reliability statistics
- **Understand** what your results mean through LLM-guided explanations
- **Explore** disagreements between coders in detail
- **Generate** comprehensive reports grounded in methodology

### Getting Started

Use the **sidebar navigation** to move through the steps:

1. **Welcome** - Enter your API key and learn about reliability
2. **Data Upload** - Upload your coded data files
3. **Configure Variables** - Set up your variables and measurement levels
4. **Run Analysis** - Compute all reliability statistics
5. **Disagreements** - Explore per-row disagreements
6. **Results** - View comprehensive dashboard
7. **Report** - Generate educational report

---
*Navigate using the pages in the sidebar →*
""")

# Render sidebar components
render_progress_indicator()
render_sidebar_chat()
