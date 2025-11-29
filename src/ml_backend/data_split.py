"""
A simple data splitting utility.

A walk forward splitting function that divides a DataFrame into

three parts: training, validation, and testing sets based on

- The training set always starts from `train_start` to the beginning of the validation period.
- The validation set spans `val_years` years starting from `val_start`.
- The testing set spans `test_years` years starting immediately after the validation period.

In other words,
- Training set: from `train_start` to `val_start - 1`, expand 1 year at a time.
- Validation set: from `val_start` to `val_start + val_years - 1`, shifting forward.
- Testing set: from `val_start + val_years` to `val_start + val_years + test_years - 1`, shifting forward.
"""

import json
import pandas as pd


# Configuration parameters
with open("configs.json", "r") as f:  # type: ignore
    config = json.load(f)["train_iteration"]
    val_years = config["val_years"]


def split_data(df: pd.DataFrame,
               predict_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and testing sets based on the provided year parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a multi-index ["date", "permno"].
    predict_year : int
        the year for which predictions are to be made, example: 2009

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the training, validation, and testing DataFrames.
    """
    test_start_date = pd.to_datetime(f"{predict_year}-01-01")
    test_end_date = pd.to_datetime(f"{predict_year}-12-31")
    
    val_start_date = test_start_date - pd.DateOffset(years=val_years)
    val_end_date = pd.to_datetime(test_start_date) - pd.DateOffset(days=1)
    
    train_end_date = val_start_date - pd.DateOffset(days=1)

    # train: df starts from the earliest date up to train_end_date
    train_df = df.loc[df.index.get_level_values("date") <= train_end_date].copy()
    # val: df from val_start_date to val_end_date
    val_df = df.loc[(df.index.get_level_values("date") >= val_start_date) &
                    (df.index.get_level_values("date") <= val_end_date)].copy()
    # test: df from test_start_date to test_end_date
    test_df = df.loc[(df.index.get_level_values("date") >= test_start_date) &
                     (df.index.get_level_values("date") <= test_end_date)].copy()
    return train_df, val_df, test_df
    
