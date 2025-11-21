import pandas as pd
from typing import List, Dict


class RollingWindowSplitter:
    """
    Rolling window: expanding train window, fixed validation and test window.
    Returns DataFrames for each iteration:
      train_df, val_df, test_df
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "DateYM"):
        self.df = df.copy()
        self.date_col = date_col

        # Convert to datetime and extract year
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df["year"] = self.df[self.date_col].dt.year

        # unique sorted years
        self.years = sorted(self.df["year"].unique())

    def get_splits(
        self,
        train_years: int = 6,
        val_years: int = 3,
        test_years: int = 1
    ) -> List[Dict[str, pd.DataFrame]]:

        splits = []
        total_years = len(self.years)

        max_start = total_years - (train_years + val_years + test_years) + 1

        for start_idx in range(max_start):

            # Define year boundaries (just like professor's 0-30-40-42)
            start_train_year = self.years[start_idx]
            end_train_year   = self.years[start_idx + train_years - 1]

            end_val_year     = self.years[start_idx + train_years + val_years - 1]

            end_test_year    = self.years[start_idx + train_years + val_years + test_years - 1]

            # Train = [start_train_year → end_train_year]
            train_df = self.df[
                (self.df["year"] >= start_train_year) &
                (self.df["year"] <= end_train_year)
            ].copy()
            train_df["WindowType"] = "train"

            # Val = (end_train_year → end_val_year]
            val_df = self.df[
                (self.df["year"] > end_train_year) &
                (self.df["year"] <= end_val_year)
            ].copy()
            val_df["WindowType"] = "val"

            # Test = (end_val_year → end_test_year]
            test_df = self.df[
                (self.df["year"] > end_val_year) &
                (self.df["year"] <= end_test_year)
            ].copy()
            test_df["WindowType"] = "test"

            splits.append({
                "train": train_df,
                "val": val_df,
                "test": test_df
            })

        return splits
