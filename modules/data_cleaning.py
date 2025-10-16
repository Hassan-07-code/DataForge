import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.log = []  # Track cleaning steps

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        removed = before - after
        self.log.append(f"🗑️ Removed {removed} duplicate rows.")

    def handle_missing(self, strategy="mean", columns=None):
        if columns is None:
            columns = self.df.columns
        for col in columns:
            if self.df[col].isnull().sum() > 0:
                if strategy == "mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == "median" and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == "mode":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        self.log.append(f"🩹 Handled missing values using **{strategy}** for {len(columns)} column(s).")

    def encode_categoricals(self, columns=None):
        le = LabelEncoder()
        if columns is None:
            columns = self.df.select_dtypes(include=["object"]).columns
        for col in columns:
            self.df[col] = le.fit_transform(self.df[col].astype(str))
        self.log.append(f"🔡 Encoded {len(columns)} categorical column(s).")

    def scale_features(self, columns=None):
        scaler = StandardScaler()
        if columns is None:
            columns = self.df.select_dtypes(include=["float64", "int64"]).columns
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.log.append(f"📏 Scaled {len(columns)} numerical column(s).")

    def remove_columns(self, columns):
        self.df.drop(columns=columns, inplace=True, errors="ignore")
        self.log.append(f"❌ Removed {len(columns)} column(s): {', '.join(columns)}")
