"""
Preprocessing module for financial panel data.
"""

import warnings
import numpy as np
import pandas as pd

from typing import Iterable

from .utils import verify_df_structure_sorted


def handle_missing(df: pd.DataFrame, 
                   feature_cols: Iterable[str]) -> pd.DataFrame:
    """Median-impute missing values within each month (cross-sectional)."""
    df[feature_cols] = df.groupby('date')[feature_cols]\
                         .transform(lambda x: x.fillna(x.median()))
    return df


def convert_binary(df: pd.DataFrame, 
                   binary_cols: Iterable[str]) -> pd.DataFrame:
    """Convert binary variables from {0,1} to {-1, +1}."""
    df[binary_cols] = df[binary_cols].replace({0: -1, 1: 1})
    return df


def categorical_to_dummy(df: pd.DataFrame,
                          categorical_cols: Iterable[str]) -> pd.DataFrame:
     """Convert categorical variables to dummy/indicator variables."""
     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
     return df


def rank_transform(df: pd.DataFrame, 
                   continuous_cols: Iterable[str]) -> pd.DataFrame:
    """
    GKX (2020) rank transformation:
    - Rank each feature cross-sectionally per month
    - Convert to percentiles [0,1]
    - Map to [-1,1]
    """
    def rank_to_unit(x):
        return x.rank(method='first', pct=True) * 2 - 1

    df[continuous_cols] = df.groupby('date')[continuous_cols].transform(rank_to_unit)
    return df


def shift_target(df: pd.DataFrame, 
                 target_col: str='retadj') -> pd.DataFrame:
    """Shift target variable to align with features."""
    df[target_col] = df.groupby(level='permno')[target_col].shift(-1)
    df.rename(columns={target_col: f"{target_col}_next"}, inplace=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Steps:

    0. Verify df and identify columns
      - target column: "retadj"
      - feature columns: all columns except target
      - binary columns: features with only {0,1} values
      - categorical columns: < 10 unique values
      - continuous columns: feature - binary - categorical
    1. Handle missing values (median-impute cross-sectionally)
    2. Convert binary variables from {0,1} to {-1,+1}
    3. Convert categorical variables to dummy/indicator variables
    4. Rank-transform features cross-sectionally per month (GKX 2020)
    5. Shift target variable to align with features
    6. fill any remaining missing values with 0
    
    ignore warnings: RuntimeWarning: Mean of empty slice
    """
    # Warnings filter
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    # Step 0: Verify df structure
    if not verify_df_structure_sorted(df):
        raise ValueError("DataFrame must have a multi-index with ['date', 'permno'] and be sorted.")

    target_col = "retadj"
    feature_cols = df.columns.difference([target_col]).tolist()
    binary_cols = [col for col in feature_cols if df.loc[:, col].dropna().isin([0,1]).all()]
    categorical_cols = [col for col in feature_cols if df[col].nunique() < 10]
    continuous_cols = list(set(feature_cols) - set(binary_cols) - set(categorical_cols))

    # Step 1: Handle missing values
    print("Preprocessing: Handling missing values...")
    df = handle_missing(df, feature_cols)
    # Step 2: Convert binary variables
    print("Preprocessing: Converting binary variables...")
    df = convert_binary(df, binary_cols)
    # Step 3: Convert categorical variables to dummies
    print("Preprocessing: Converting categorical variables to dummies...")
    df = categorical_to_dummy(df, categorical_cols)
    # Step 4: Rank-transform features
    print("Preprocessing: Rank-transforming continuous variables...")
    df = rank_transform(df, continuous_cols)
    # Step 5: Shift target variable
    print("Preprocessing: Shifting target variable...")
    df = shift_target(df, target_col)
    # Step 6: Fill any remaining missing values with 0
    print("Preprocessing: Filling remaining missing values with 0...")
    df = df.fillna(0)
    
    # modify all float64 to float32 to save memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    # bool columns to int8
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(np.int8)
    return df.sort_index()

