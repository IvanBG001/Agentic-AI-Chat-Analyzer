import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_duplicates(self) -> pd.DataFrame:
        # Only use columns that are hashable
        cols_to_check = [col for col in self.df.columns if self.df[col].apply(lambda x: isinstance(x, list)).sum() == 0]
        return self.df.drop_duplicates(subset=cols_to_check)

    def handle_missing_values(self) -> pd.DataFrame:
        self.df["message"].fillna("", inplace=True)
        self.df.fillna("Unknown", inplace=True)
        return self.df

    def correct_dtypes(self) -> pd.DataFrame:
        self.df["agent"] = self.df["agent"].astype(str)
        self.df["sentiment"] = self.df["sentiment"].astype(str)
        self.df["message"] = self.df["message"].astype(str)
        return self.df

    def clean_all(self) -> pd.DataFrame:
        self.df = self.remove_duplicates()
        self.df = self.handle_missing_values()
        self.df = self.correct_dtypes()
        return self.df
