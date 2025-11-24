"""
Utility functions for various common tasks.
"""

import pandas as pd


def verify_df_structure_sorted(df: pd.DataFrame) -> bool:
    """
    Verify that the DataFrame 

    - has a multi-index with 'date' and 'permno', 
    - sorted
    - duplicate-free.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to verify.

    Returns
    -------
    bool
        True if the DataFrame has the correct structure, False otherwise.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return False
    expected_levels = ['date', 'permno']

    if list(df.index.names) != expected_levels:
        return False
    if not df.index.is_monotonic_increasing:
        return False
    if df.index.duplicated().any():
        return False
    return True

