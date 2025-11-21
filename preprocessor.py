import pandas as pd
import numpy as np

def handle_missing(df, feature_cols):
    """
    Median-impute missing values within each month (cross-sectional).
    """
    df[feature_cols] = df.groupby('date')[feature_cols]\
                         .transform(lambda x: x.fillna(x.median()))
    return df

def convert_binary(df, binary_cols):
    df[binary_cols] = df[binary_cols].replace({0: -1, 1: 1})
    return df

def rank_transform(df, feature_cols):
    """
    GKX (2020) rank transformation:
    - Rank each feature cross-sectionally per month
    - Convert to percentiles [0,1]
    - Map to [-1,1]
    """
    def rank_to_unit(x):
        return x.rank(method='first', pct=True) * 2 - 1

    df[feature_cols] = df.groupby('date')[feature_cols].transform(rank_to_unit)
    return df

def create_target(df, return_col='excess_ret'):
    """
    Shift next-month excess return to create the prediction target.
    """
    df['target'] = df.groupby('permno')[return_col].shift(-1)
    return df.dropna(subset=['target'])

def preprocess(df, feature_cols, binary_cols=None, return_col='retadj'):
    # 1. Impute missing values
    df = handle_missing(df, feature_cols)

    # 2. Convert binary variables to {-1, +1}
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
    print(binary_cols)
    if len(binary_cols) > 0:
        df = convert_binary(df, binary_cols)

    # 3. Rank-transform continuous predictors
    df = rank_transform(df, feature_cols)

    # 4. Create next-month return target
    df = create_target(df, return_col)

    return df

