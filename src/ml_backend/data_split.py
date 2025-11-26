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
import pandas as pd


def split_data(df: pd.DataFrame,
               train_start: int,
               val_start: int,
               val_years: int,
               test_years: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and testing sets based on the provided year parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with a multi-index ["date", "permno"].
    train_start : int
        The starting year for the training set.
    val_start : int
        The starting year for the validation set.
    val_years : int
        The number of years to include in the validation set.
    test_years : int
        The number of years to include in the testing set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the training, validation, and testing DataFrames.
    """
    start_date = pd.to_datetime(f"{train_start}-01-01")
    train_end_date = pd.to_datetime(f"{val_start}-01-01") - pd.Timedelta(days=1)
    val_start_date = pd.to_datetime(f"{val_start}-01-01")
    val_end_date = pd.to_datetime(f"{val_start + val_years}-01-01") - pd.Timedelta(days=1)
    test_start_date = pd.to_datetime(f"{val_start + val_years}-01-01")
    test_end_date = pd.to_datetime(f"{val_start + val_years + test_years}-01-01") - pd.Timedelta(days=1)

    train_df = df.loc[(df.index.get_level_values("date") >= start_date) &
                      (df.index.get_level_values("date") <= train_end_date)].copy()
    val_df = df.loc[(df.index.get_level_values("date") >= val_start_date) &
                    (df.index.get_level_values("date") <= val_end_date)].copy()
    test_df = df.loc[(df.index.get_level_values("date") >= test_start_date) &
                     (df.index.get_level_values("date") <= test_end_date)].copy()
    return train_df, val_df, test_df
    