"""Step 2: Data Upload & Preview."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Upload - K's Alpha Calculator", layout="wide")

st.title("Step 2: Upload Your Data")

# Check prerequisites
if not st.session_state.app_state.get('api_key'):
    st.warning("Please enter your API key on the Welcome page first.")
    st.stop()

st.markdown("""
Upload your coded data files. The app supports several formats:

- **Multiple XLSX files**: One file per coder (recommended for your format)
- **Single CSV/XLSX**: With coder ID column
- **Long format CSV**: With unit_id, coder, variable, value columns
""")

with st.expander("Data Format Guide"):
    st.markdown("""
    ### Multiple Files (One Per Coder)

    Each coder has their own file with the same structure:
    - Rows = units (articles, cases, etc.)
    - Columns = metadata + coded variables

    ### Single File with Coder Column

    One file where each row is a coding instance:
    - unit_id column: identifies the unit
    - coder_id column: identifies who coded it
    - Other columns: coded variables

    ### Long Format

    Tidy data format with columns:
    - unit_id, coder, variable, value
    """)

st.markdown("---")

# Upload mode selection
upload_mode = st.radio(
    "Select upload mode:",
    ["Multiple files (one per coder)", "Single file"],
    index=0,
)

if upload_mode == "Multiple files (one per coder)":
    st.subheader("Upload Coder Files")

    uploaded_files = st.file_uploader(
        "Upload XLSX or CSV files (one per coder)",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Upload one file for each coder. Files should have the same structure.",
    )

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")

        # Let user name each coder
        st.subheader("Name Your Coders")

        coder_files = {}
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([1, 3])
            with col1:
                # Extract name from filename
                default_name = file.name.split('.')[0].replace('ICR_', '').replace('icr_', '')
                coder_name = st.text_input(
                    f"Coder name for {file.name}",
                    value=default_name,
                    key=f"coder_name_{i}",
                )
            with col2:
                # Preview
                file.seek(0)
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file, nrows=3)
                else:
                    df = pd.read_excel(file, nrows=3)
                st.dataframe(df, use_container_width=True, height=130)
                file.seek(0)

            coder_files[coder_name] = file

        st.session_state.app_state['raw_dataframes'] = coder_files
        st.session_state.app_state['upload_mode'] = 'multi_xlsx'

        # Show combined info
        st.markdown("---")
        st.subheader("Data Summary")

        # Get column info from first file
        first_file = list(coder_files.values())[0]
        first_file.seek(0)
        if first_file.name.endswith('.csv'):
            df_sample = pd.read_csv(first_file)
        else:
            df_sample = pd.read_excel(first_file)
        first_file.seek(0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Coders", len(coder_files))
            st.metric("Number of Rows (units)", len(df_sample))
        with col2:
            st.metric("Number of Columns", len(df_sample.columns))

        st.markdown("**Columns found:**")
        st.write(df_sample.columns.tolist())

        # Store column info
        st.session_state.app_state['columns'] = df_sample.columns.tolist()
        st.session_state.app_state['n_rows'] = len(df_sample)

else:
    st.subheader("Upload Single File")

    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX file",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=False,
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        # Preview
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.session_state.app_state['raw_dataframes'] = {'single': uploaded_file}
        st.session_state.app_state['upload_mode'] = 'single'
        st.session_state.app_state['columns'] = df.columns.tolist()
        st.session_state.app_state['n_rows'] = len(df)

# LLM Analysis (if data uploaded and LLM available)
if st.session_state.app_state.get('raw_dataframes') and st.session_state.app_state.get('llm_client'):
    st.markdown("---")

    if st.button("Analyze Data Structure with AI"):
        with st.spinner("Analyzing data structure..."):
            try:
                # Get sample data for analysis
                files = st.session_state.app_state['raw_dataframes']
                first_file = list(files.values())[0]
                first_file.seek(0)

                if first_file.name.endswith('.csv'):
                    df = pd.read_csv(first_file, nrows=10)
                else:
                    df = pd.read_excel(first_file, nrows=10)
                first_file.seek(0)

                from core.llm_client import analyze_data_structure

                analysis = analyze_data_structure(
                    st.session_state.app_state['llm_client'],
                    df.to_string(),
                    df.columns.tolist(),
                )

                st.session_state.app_state['data_analysis'] = analysis

                st.success("Analysis complete!")

                st.subheader("AI Analysis Results")
                st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")
                st.markdown(f"**Detected Format:** {analysis.get('data_format', 'N/A')}")

                if analysis.get('coded_variables'):
                    st.markdown("**Detected Variables:**")
                    for var in analysis['coded_variables']:
                        st.markdown(f"- {var['name']} ({var['suggested_level']}): {var.get('reason', '')}")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Navigation
st.markdown("---")
if st.session_state.app_state.get('raw_dataframes'):
    st.session_state.app_state['current_step'] = max(st.session_state.app_state.get('current_step', 1), 2)
    st.info("Data uploaded! Navigate to **3. Configure Variables** to continue.")
