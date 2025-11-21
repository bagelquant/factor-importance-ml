import pandas as pd
import numpy as np


def prepare_data(df, date_col="date"):
    """
    Ensures date column is datetime and dataset is sorted chronologically.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    return df


def train_test_split(df, split_date="2017-01-01", date_col="date"):
    """
    Simple chronological train–test split for firm-month panel data.
    """
    train = df[df[date_col] < split_date]
    test = df[df[date_col] >= split_date]
    return train, test


def expanding_window_cv(df, start_year=2000, end_year=2023, date_col="date"):
    """
    Generates expanding-window time-series CV folds for cross-sectional return prediction.
    Returns a list of dictionaries: {"year": int, "train": df, "test": df}.
    """
    df = df.sort_values(date_col)
    df[date_col] = pd.to_datetime(df[date_col])

    cv_splits = []

    for year in range(start_year + 6, end_year):
        test_start = f"{year}-01-01"
        test_end = f"{year+1}-01-01"

        train_df = df[df[date_col] < test_start]
        test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]

        if len(test_df) == 0:
            continue

        cv_splits.append({
            "year": year,
            "train": train_df,
            "test": test_df
        })

    return cv_splits
