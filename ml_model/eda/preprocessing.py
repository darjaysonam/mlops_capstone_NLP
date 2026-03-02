import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:

    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column

    # ----------------------------------------
    # 1. Remove Duplicates
    # ----------------------------------------
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self.df

    # ----------------------------------------
    # 2. Missing Value Imputation
    # ----------------------------------------
    def impute_missing(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns

        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        self.df[numeric_cols] = num_imputer.fit_transform(self.df[numeric_cols])
        self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])

        return self.df

    # ----------------------------------------
    # 3. Encoding Categorical Variables
    # ----------------------------------------
    def encode_categorical(self):
        for col in self.df.select_dtypes(include="object").columns:
            if col != self.target_column:
                self.df[col] = LabelEncoder().fit_transform(self.df[col])
        return self.df

    # ----------------------------------------
    # 4. Feature Scaling
    # ----------------------------------------
    def scale_features(self):
        features = self.df.drop(columns=[self.target_column])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        self.df[features.columns] = scaled
        return self.df