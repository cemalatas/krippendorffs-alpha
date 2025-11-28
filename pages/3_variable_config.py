"""Step 3: Variable Configuration."""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Configure Variables - K's Alpha Calculator", layout="wide")

st.title("Step 3: Configure Your Variables")

# Check prerequisites
if not st.session_state.app_state.get('raw_dataframes'):
    st.warning("Please upload your data on the Data Upload page first.")
    st.stop()

st.markdown("""
Configure which columns contain your coded variables and specify the measurement level
for each variable.
""")

with st.expander("Understanding Measurement Levels"):
    st.markdown("""
    | Level | Description | Example |
    |-------|-------------|---------|
    | **Nominal** | Categories with no inherent order | positive/negative/neutral; yes/no |
    | **Ordinal** | Ordered categories, intervals not equal | 1-3 rating scale; low/medium/high |
    | **Interval** | Equal intervals, no true zero | Temperature in Celsius |
    | **Ratio** | Equal intervals with true zero | Counts; percentages |

    **Tip:** Most content analysis variables are **nominal** (categories) or **ordinal** (rating scales).
    """)

st.markdown("---")

# Get column info
columns = st.session_state.app_state.get('columns', [])
upload_mode = st.session_state.app_state.get('upload_mode', 'multi_xlsx')

# For multi-file upload, configure column indices
if upload_mode == 'multi_xlsx':
    st.subheader("Column Configuration for Multi-File Upload")

    st.markdown("""
    Your files appear to have metadata columns followed by coded variables.
    Please specify which columns contain what.
    """)

    # Load preview data
    files = st.session_state.app_state['raw_dataframes']
    first_file = list(files.values())[0]
    first_file.seek(0)
    if first_file.name.endswith('.csv'):
        df_preview = pd.read_csv(first_file, nrows=5)
    else:
        df_preview = pd.read_excel(first_file, nrows=5)
    first_file.seek(0)

    # Unit ID Column selection - NO DEFAULT
    st.subheader("1. Select Unit ID Column")
    st.caption("This column identifies each unit (article, case, etc.) being coded. You must select one.")

    unit_id_col = st.selectbox(
        "Unit ID Column",
        options=[None] + list(range(len(columns))),
        format_func=lambda x: "-- Select a column --" if x is None else f"{x}: {columns[x]}",
        index=0,  # Default to None (first option)
        help="Select the column that uniquely identifies each row/unit",
    )

    if unit_id_col is not None:
        st.markdown("**Preview of selected unit ID column:**")
        st.write(df_preview.iloc[:, unit_id_col].tolist())
    else:
        st.warning("Please select a Unit ID column to continue.")

    st.markdown("---")
    st.subheader("2. Select Coded Variables")

    st.markdown("Select the columns that contain coded values (variables you want to analyze):")

    # Use multiselect for variable selection
    all_column_options = [(i, f"{i}: {columns[i]}") for i in range(len(columns))]

    selected_var_indices = st.multiselect(
        "Select variable columns",
        options=[i for i, _ in all_column_options],
        format_func=lambda x: f"{x}: {columns[x]}",
        help="Select all columns that contain coded variables",
    )

    variable_indices = selected_var_indices
    variable_names = [columns[i] for i in variable_indices] if variable_indices else []

    if not variable_indices:
        st.info("Tip: For your data format, variables are typically in columns 7-30. Select them above.")

    st.markdown(f"**Selected {len(variable_names)} variables:**")
    st.write(variable_names)

    # Measurement level configuration
    st.markdown("---")
    st.subheader("Configure Measurement Levels")

    st.markdown("Set the measurement level for each variable:")

    variable_levels = {}

    # Allow bulk setting
    bulk_level = st.selectbox(
        "Set all variables to:",
        options=["(individual)", "nominal", "ordinal", "interval", "ratio"],
        index=0,
    )

    if bulk_level != "(individual)":
        for var in variable_names:
            variable_levels[var] = bulk_level
        st.success(f"All variables set to {bulk_level}")
    else:
        # Individual configuration
        cols = st.columns(3)
        for i, var in enumerate(variable_names):
            with cols[i % 3]:
                # Default based on variable name
                default_level = "nominal"
                if "1-3" in var or "rating" in var.lower() or "scale" in var.lower():
                    default_level = "ordinal"

                level = st.selectbox(
                    f"{var}",
                    options=["nominal", "ordinal", "interval", "ratio"],
                    index=["nominal", "ordinal", "interval", "ratio"].index(default_level),
                    key=f"level_{var}",
                )
                variable_levels[var] = level

    # Store configuration
    st.session_state.app_state['variable_columns'] = variable_indices
    st.session_state.app_state['variable_names'] = variable_names
    st.session_state.app_state['variable_levels'] = variable_levels
    st.session_state.app_state['unit_id_column'] = unit_id_col

else:
    # Single file configuration
    st.subheader("Column Configuration for Single File")

    coder_column = st.selectbox(
        "Coder ID Column",
        options=columns,
        help="Column that identifies which coder made each coding",
    )

    unit_id_column = st.selectbox(
        "Unit ID Column",
        options=columns,
        help="Column that identifies each unit being coded",
    )

    available_vars = [c for c in columns if c not in [coder_column, unit_id_column]]

    variable_names = st.multiselect(
        "Select Variables to Analyze",
        options=available_vars,
        default=available_vars[:10] if len(available_vars) > 10 else available_vars,
    )

    # Measurement levels
    st.subheader("Configure Measurement Levels")

    variable_levels = {}
    cols = st.columns(3)
    for i, var in enumerate(variable_names):
        with cols[i % 3]:
            level = st.selectbox(
                f"{var}",
                options=["nominal", "ordinal", "interval", "ratio"],
                key=f"level_{var}",
            )
            variable_levels[var] = level

    st.session_state.app_state['coder_column'] = coder_column
    st.session_state.app_state['unit_id_column'] = unit_id_column
    st.session_state.app_state['variable_names'] = variable_names
    st.session_state.app_state['variable_levels'] = variable_levels

# Transform data button
st.markdown("---")
st.subheader("Confirm Configuration")

# Validation
can_transform = True
if upload_mode == 'multi_xlsx':
    if unit_id_col is None:
        st.error("Please select a Unit ID column before continuing.")
        can_transform = False
    if not variable_indices:
        st.error("Please select at least one variable column.")
        can_transform = False

if st.button("Transform Data", type="primary", disabled=not can_transform):
    with st.spinner("Transforming data to canonical format..."):
        try:
            from core.data_transformer import transform_to_coder_data

            files = st.session_state.app_state['raw_dataframes']

            if upload_mode == 'multi_xlsx':
                config = {
                    'format': 'multi_xlsx',
                    'variable_columns': st.session_state.app_state['variable_columns'],
                    'variable_names': st.session_state.app_state['variable_names'],
                    'variable_levels': st.session_state.app_state['variable_levels'],
                    'unit_id_column': st.session_state.app_state['unit_id_column'],
                }
            else:
                config = {
                    'format': 'single_csv' if list(files.values())[0].name.endswith('.csv') else 'single_xlsx',
                    'coder_column': st.session_state.app_state['coder_column'],
                    'unit_id_column': st.session_state.app_state['unit_id_column'],
                    'variable_columns': st.session_state.app_state['variable_names'],
                    'variable_levels': st.session_state.app_state['variable_levels'],
                }

            coder_data = transform_to_coder_data(files, config)

            st.session_state.app_state['coder_data'] = coder_data
            st.session_state.app_state['current_step'] = max(st.session_state.app_state.get('current_step', 1), 3)

            st.success("Data transformed successfully!")

            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coders", coder_data.n_coders)
            with col2:
                st.metric("Units", coder_data.n_units)
            with col3:
                st.metric("Variables", len(coder_data.variables))

            st.markdown("**Coders:** " + ", ".join(coder_data.coder_names))
            st.markdown("**Variables:** " + ", ".join(coder_data.variables[:10]) + ("..." if len(coder_data.variables) > 10 else ""))

            st.info("Navigate to **4. Run Analysis** to compute reliability statistics.")

        except Exception as e:
            st.error(f"Transformation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
