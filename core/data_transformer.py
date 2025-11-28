"""Data transformation utilities for loading and standardizing coder data."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .models import CoderData


def detect_format(
    files: List[Any],  # Can be file paths or uploaded file objects
) -> str:
    """Detect the input format from uploaded files.

    Args:
        files: List of file objects or paths

    Returns:
        Format string: "multi_xlsx", "single_csv", "single_xlsx", "long_csv"
    """
    if len(files) > 1:
        # Multiple files = one per coder
        return "multi_xlsx"

    file = files[0]
    name = file.name if hasattr(file, 'name') else str(file)

    if name.endswith('.xlsx') or name.endswith('.xls'):
        return "single_xlsx"
    elif name.endswith('.csv'):
        # Read first few rows to detect format
        if hasattr(file, 'seek'):
            file.seek(0)
        df = pd.read_csv(file, nrows=5)
        if hasattr(file, 'seek'):
            file.seek(0)

        # Check if it's long format
        if {'unit_id', 'coder', 'variable', 'value'}.issubset(set(df.columns.str.lower())):
            return "long_csv"
        return "single_csv"

    raise ValueError(f"Unsupported file format: {name}")


def load_multi_xlsx(
    files: Dict[str, Any],  # {coder_name: file_object}
    variable_columns: List[int],
    variable_names: List[str],
    variable_levels: Dict[str, str],
    unit_id_column: int = 0,
) -> CoderData:
    """Load data from multiple XLSX files (one per coder).

    This matches the format used in the base icr_krippendorff_compute.py

    Args:
        files: Dict mapping coder name to file object
        variable_columns: List of column indices containing coded variables
        variable_names: List of variable names (matching variable_columns order)
        variable_levels: Dict mapping variable names to measurement levels
        unit_id_column: Column index for unit identifiers

    Returns:
        CoderData instance
    """
    coder_names = list(files.keys())
    frames: Dict[str, pd.DataFrame] = {}

    # Load each coder's data
    for coder, file_obj in files.items():
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        df_raw = pd.read_excel(file_obj)

        # Extract unit IDs from first coder only
        if coder == coder_names[0]:
            unit_ids = df_raw.iloc[:, unit_id_column].astype(str).tolist()

        # Extract coded columns and rename
        rename_map = {df_raw.columns[idx]: var_name
                      for idx, var_name in zip(variable_columns, variable_names)}
        frames[coder] = df_raw.rename(columns=rename_map)[variable_names]

    # Build data matrices for each variable
    n_units = len(unit_ids)
    n_coders = len(coder_names)

    data: Dict[str, np.ndarray] = {}
    for var in variable_names:
        matrix = np.full((n_coders, n_units), np.nan)
        for coder_idx, coder in enumerate(coder_names):
            values = frames[coder][var].to_numpy(dtype=float)
            matrix[coder_idx, :len(values)] = values
        data[var] = matrix

    return CoderData(
        coder_names=coder_names,
        unit_ids=unit_ids,
        variable_levels=variable_levels,
        data=data,
    )


def load_single_xlsx_wide(
    file: Any,
    coder_column: str,
    unit_id_column: str,
    variable_columns: List[str],
    variable_levels: Dict[str, str],
) -> CoderData:
    """Load data from single XLSX with rows containing coder IDs.

    Format: Each row is a (unit, coder) pair with coded values in columns.

    Args:
        file: File object
        coder_column: Column name for coder identifiers
        unit_id_column: Column name for unit identifiers
        variable_columns: List of column names containing coded variables
        variable_levels: Dict mapping variable names to measurement levels

    Returns:
        CoderData instance
    """
    if hasattr(file, 'seek'):
        file.seek(0)
    df = pd.read_excel(file)

    return _transform_long_to_coder_data(
        df, coder_column, unit_id_column, variable_columns, variable_levels
    )


def load_single_csv(
    file: Any,
    coder_column: str,
    unit_id_column: str,
    variable_columns: List[str],
    variable_levels: Dict[str, str],
) -> CoderData:
    """Load data from single CSV with rows containing coder IDs.

    Args:
        file: File object
        coder_column: Column name for coder identifiers
        unit_id_column: Column name for unit identifiers
        variable_columns: List of column names containing coded variables
        variable_levels: Dict mapping variable names to measurement levels

    Returns:
        CoderData instance
    """
    if hasattr(file, 'seek'):
        file.seek(0)
    df = pd.read_csv(file)

    return _transform_long_to_coder_data(
        df, coder_column, unit_id_column, variable_columns, variable_levels
    )


def load_long_csv(
    file: Any,
    variable_levels: Dict[str, str],
) -> CoderData:
    """Load data from long-format CSV.

    Expected columns: unit_id, coder, variable, value

    Args:
        file: File object
        variable_levels: Dict mapping variable names to measurement levels

    Returns:
        CoderData instance
    """
    if hasattr(file, 'seek'):
        file.seek(0)
    df = pd.read_csv(file)

    # Standardize column names
    df.columns = df.columns.str.lower()

    # Pivot to wide format first
    pivot = df.pivot_table(
        index=['unit_id', 'coder'],
        columns='variable',
        values='value',
        aggfunc='first'
    ).reset_index()

    variable_columns = [c for c in pivot.columns if c not in ['unit_id', 'coder']]

    return _transform_long_to_coder_data(
        pivot, 'coder', 'unit_id', variable_columns, variable_levels
    )


def _transform_long_to_coder_data(
    df: pd.DataFrame,
    coder_column: str,
    unit_id_column: str,
    variable_columns: List[str],
    variable_levels: Dict[str, str],
) -> CoderData:
    """Transform a DataFrame with coder/unit columns to CoderData.

    Args:
        df: Input DataFrame
        coder_column: Column name for coder identifiers
        unit_id_column: Column name for unit identifiers
        variable_columns: List of variable column names
        variable_levels: Dict mapping variable names to measurement levels

    Returns:
        CoderData instance
    """
    coder_names = sorted(df[coder_column].unique().tolist())
    unit_ids = sorted(df[unit_id_column].unique().astype(str).tolist())

    n_coders = len(coder_names)
    n_units = len(unit_ids)

    # Create mapping for fast lookup
    coder_to_idx = {c: i for i, c in enumerate(coder_names)}
    unit_to_idx = {u: i for i, u in enumerate(unit_ids)}

    # Build data matrices
    data: Dict[str, np.ndarray] = {}

    for var in variable_columns:
        matrix = np.full((n_coders, n_units), np.nan)

        for _, row in df.iterrows():
            coder_idx = coder_to_idx[row[coder_column]]
            unit_idx = unit_to_idx[str(row[unit_id_column])]
            value = row[var]

            if pd.notna(value):
                try:
                    matrix[coder_idx, unit_idx] = float(value)
                except (ValueError, TypeError):
                    # Handle non-numeric values by using hash
                    matrix[coder_idx, unit_idx] = hash(str(value)) % 1000

        data[var] = matrix

    # Fill in any missing variable levels with 'nominal'
    full_levels = {var: variable_levels.get(var, 'nominal') for var in variable_columns}

    return CoderData(
        coder_names=coder_names,
        unit_ids=unit_ids,
        variable_levels=full_levels,
        data=data,
    )


def transform_to_coder_data(
    files: Union[Dict[str, Any], List[Any], Any],
    config: Dict[str, Any],
) -> CoderData:
    """Main entry point for transforming uploaded files to CoderData.

    Args:
        files: Uploaded file(s) - dict for multi-file, list or single for others
        config: Configuration dict with:
            - format: "multi_xlsx", "single_xlsx", "single_csv", "long_csv"
            - variable_columns: List of column indices or names
            - variable_names: List of variable names (for multi_xlsx)
            - variable_levels: Dict of variable -> measurement level
            - coder_column: Column name for coder ID (if applicable)
            - unit_id_column: Column name/index for unit ID

    Returns:
        CoderData instance
    """
    fmt = config.get('format', 'multi_xlsx')

    if fmt == 'multi_xlsx':
        return load_multi_xlsx(
            files=files if isinstance(files, dict) else {f"Coder_{i}": f for i, f in enumerate(files)},
            variable_columns=config['variable_columns'],
            variable_names=config['variable_names'],
            variable_levels=config['variable_levels'],
            unit_id_column=config.get('unit_id_column', 0),
        )
    elif fmt == 'single_xlsx':
        return load_single_xlsx_wide(
            file=files[0] if isinstance(files, list) else files,
            coder_column=config['coder_column'],
            unit_id_column=config['unit_id_column'],
            variable_columns=config['variable_columns'],
            variable_levels=config['variable_levels'],
        )
    elif fmt == 'single_csv':
        return load_single_csv(
            file=files[0] if isinstance(files, list) else files,
            coder_column=config['coder_column'],
            unit_id_column=config['unit_id_column'],
            variable_columns=config['variable_columns'],
            variable_levels=config['variable_levels'],
        )
    elif fmt == 'long_csv':
        return load_long_csv(
            file=files[0] if isinstance(files, list) else files,
            variable_levels=config['variable_levels'],
        )
    else:
        raise ValueError(f"Unknown format: {fmt}")


def preview_dataframe(file: Any, n_rows: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """Load and preview a data file.

    Args:
        file: File object
        n_rows: Number of rows to preview

    Returns:
        Tuple of (preview DataFrame, list of column names)
    """
    if hasattr(file, 'seek'):
        file.seek(0)

    name = file.name if hasattr(file, 'name') else str(file)

    if name.endswith('.xlsx') or name.endswith('.xls'):
        df = pd.read_excel(file, nrows=n_rows)
    else:
        df = pd.read_csv(file, nrows=n_rows)

    if hasattr(file, 'seek'):
        file.seek(0)

    return df, df.columns.tolist()


def get_unique_values(file: Any, column: Union[str, int]) -> List[Any]:
    """Get unique values from a column.

    Args:
        file: File object
        column: Column name or index

    Returns:
        List of unique values
    """
    if hasattr(file, 'seek'):
        file.seek(0)

    name = file.name if hasattr(file, 'name') else str(file)

    if name.endswith('.xlsx') or name.endswith('.xls'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    if hasattr(file, 'seek'):
        file.seek(0)

    if isinstance(column, int):
        return df.iloc[:, column].dropna().unique().tolist()
    return df[column].dropna().unique().tolist()
