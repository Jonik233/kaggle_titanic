import numpy as np
import pandas as pd
from typing import Tuple, List, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class TransformerPipeline(BaseEstimator, TransformerMixin):

    COLS_TO_ENCODE = ["Pclass", "Embarked", "Family_size"] # Sex and Name_title are encoded separately from these

    COLS_TO_DROP = ["PassengerId", "Name", "Cabin", "Ticket", "Parch", "SibSp"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy="mean")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.universal_encoder = None

    def fit(self, df: pd.DataFrame):
        """
        Transformer fit method
        :param df: pd.DataFrame
        :return: self object
        """
        return self

    def __encode(self, df: pd.DataFrame) -> pd.DataFrame:

        # Encoding categorical attribute 'Sex'
        df.loc[:, "Sex"] = df["Sex"].apply(lambda sex: 1 if sex == "male" else 0)

        # Applying universal encoder to COLS_TO_ENCODE
        if self.universal_encoder is None:
            drop_categories = [df[col].value_counts().idxmin() for col in self.COLS_TO_ENCODE]
            self.universal_encoder = OneHotEncoder(drop=drop_categories, sparse_output=False, dtype=np.int64)
            self.universal_encoder.fit(df[self.COLS_TO_ENCODE])

        cols_encoded_arr = self.universal_encoder.transform(df[self.COLS_TO_ENCODE])
        cols_encoded_df = pd.DataFrame(cols_encoded_arr, columns=self.universal_encoder.get_feature_names_out(self.COLS_TO_ENCODE), dtype=np.int64)

        df = df.drop(columns=[*self.COLS_TO_ENCODE]).reset_index(drop=True)
        df = pd.concat([df, cols_encoded_df], axis=1)

        return df

    def __create_new_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Family_size"] = df["Parch"] + df["SibSp"]
        return df

    def __bin_family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Family_size"] = pd.cut(
            df["Family_size"],
            bins=[-1, 0, 3, 5, 10],
            labels=["Alone", "Small", "Medium", "Large"]
        )

        return df

    def __fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        self.num_imputer.fit(df[["Age", "Fare"]]) if not hasattr(self.num_imputer, "statistics_") else None
        self.cat_imputer.fit(df[["Embarked"]]) if not hasattr(self.cat_imputer, "statistics_") else None

        df[["Age", "Fare"]] = self.num_imputer.transform(df[["Age", "Fare"]])
        df["Embarked"] = self.cat_imputer.transform(df[["Embarked"]]).ravel()

        return df

    def __scale(self, df: pd.DataFrame) -> pd.DataFrame:
        self.scaler.fit(df[["Age", "Fare"]]) if not hasattr(self.scaler, 'mean_') else None
        df[["Age", "Fare"]] = self.scaler.transform(df[["Age", "Fare"]])
        df["Fare"] = np.log10(df["Fare"] + 1)
        return df

    def __drop(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns from df specified in COLS_TO_DROP
        :param df: pd.DataFrame
        :return: data frame
        """
        data = df.drop(columns=self.COLS_TO_DROP)
        return data

    def _get_transformations(self) -> List[Callable]:
        """
        Forms and returns a list of transformations
        :return: List[Callable]
        """
        return [
            self.__fill_na,
            self.__create_new_cols,
            self.__bin_family_size,
            self.__encode,
            self.__scale,
            self.__drop,
        ]

    def transform(
        self,
        df: pd.DataFrame,
        split_attr=False,
        label: str = "Survived",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies transformations and returns prepared inputs and/or labels
        :param df: Data Frame with input data
        :param split_attr: in case true - splits Data Frame into inputs and labels
        :param label: naming of the target column
        :return: Tuple of inputs and/or labels
        """

        # Initializing labels var in case split_attr is True
        labels = None

        # Applying transformations
        for fn in self._get_transformations():
            df = fn(df)

        if split_attr:
            labels = df[[label]].values.reshape(-1, )
            # Drop label if splitting attributes
            df = df.drop(columns=[label])

        inputs = df.values.astype(np.float32)
        result = (inputs, labels) if split_attr else inputs

        return result