import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import dotenv_values
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import (
    COLS_TO_ENCODE,
    PCLASS_CATEGORIES_TO_ENCODE,
    EMBARKED_CATEGORIES_TO_ENCODE,
    FAMILY_SIZE_CATEGORIES_TO_ENCODE,
    TITLE_CATEGORIES_TO_ENCODE,
    COLS_TO_DROP,
    CATEGORICAL_COLS_TO_FILL,
    NUMERICAL_COLS_TO_FILL,
    FAMILY_SIZE_BINS,
    FAMILY_SIZE_LABELS,
    STANDARD_SCALE_COLS,
    LOG_SCALE_COLS,
    ENV_FILE_PATH,
)

np.random.seed(42)


def fill_na(df: pd.DataFrame) -> pd.DataFrame:

    # Loading env config and imputers paths
    env_config = dotenv_values(ENV_FILE_PATH)
    num_imputer_path = Path(env_config["NUMERIC_IMPUTER_DUMP_PATH"])
    cat_imputer_path = Path(env_config["CATEGORICAL_IMPUTER_DUMP_PATH"])

    if num_imputer_path.exists():
        # Loading numerical imputer
        num_imputer = joblib.load(num_imputer_path)
    else:
        print(f"\nNumerical imputer not found in: {num_imputer_path}")
        print("Creating new numerical imputer...")

        # Creating new numerical imputer
        num_imputer = SimpleImputer(strategy="mean")
        num_imputer.fit(df[NUMERICAL_COLS_TO_FILL])

        # Loading new numerical imputer into the dump
        print("Loading new numerical imputer into the dump...")
        joblib.dump(num_imputer, num_imputer_path)

    if cat_imputer_path.exists():
        # Loading categorical imputer
        cat_imputer = joblib.load(cat_imputer_path)
    else:
        print(f"\nCategorical imputer not found in: {cat_imputer_path}")
        print("Creating new categorical imputer...")

        # Creating new categorical imputer
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer.fit(df[CATEGORICAL_COLS_TO_FILL])

        # Loading new categorical imputer into the dump
        print("Loading new categorical imputer into the dump...")
        joblib.dump(cat_imputer, cat_imputer_path)

    # Applying imputers
    df[NUMERICAL_COLS_TO_FILL] = num_imputer.transform(df[NUMERICAL_COLS_TO_FILL])
    df[CATEGORICAL_COLS_TO_FILL] = cat_imputer.transform(df[CATEGORICAL_COLS_TO_FILL])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:

    # Encoding categorical column 'Sex'
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Load config and encoder path
    env_config = dotenv_values(ENV_FILE_PATH)
    encoder_path = Path(env_config["UNIVERSAL_ENCODER_DUMP_PATH"])

    if encoder_path.exists():
        # Loading universal encoder
        universal_encoder = joblib.load(encoder_path)
    else:
        print(f"\nUniversal encoder not found in {encoder_path}")
        print("Creating new encoder...")

        # Creating new encoder
        universal_encoder = OneHotEncoder(
            categories=[
                PCLASS_CATEGORIES_TO_ENCODE,
                EMBARKED_CATEGORIES_TO_ENCODE,
                FAMILY_SIZE_CATEGORIES_TO_ENCODE,
                TITLE_CATEGORIES_TO_ENCODE,
            ],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.int64,
        )
        universal_encoder.fit(df[COLS_TO_ENCODE])

        # Loading new encoder into the dump
        print(f"Saving to encoder to {encoder_path}")
        joblib.dump(universal_encoder, encoder_path)

    # Applying universal encoder
    encoded_arr = universal_encoder.transform(df[COLS_TO_ENCODE])
    encoded_df = pd.DataFrame(
        data=encoded_arr,
        columns=universal_encoder.get_feature_names_out(COLS_TO_ENCODE),
        index=df.index,
        dtype=np.int64,
    )

    # Replacing unencoded columns with encoded
    df = df.drop(columns=COLS_TO_ENCODE)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def scale(df: pd.DataFrame) -> pd.DataFrame:
    # Loading env config and scaler path
    env_config = dotenv_values(ENV_FILE_PATH)
    scaler_path = Path(env_config["SCALER_DUMP_PATH"])

    if scaler_path.exists():
        # Loading scaler
        scaler = joblib.load(scaler_path)
    else:
        print(f"\nScaler not found in {scaler_path}")
        print("Creating new scaler...")

        # Creating new scaler
        scaler = StandardScaler()
        scaler.fit(df[STANDARD_SCALE_COLS])

        # Loading new scaler into the dump
        print("Loading scaler into the dump...")
        joblib.dump(scaler, scaler_path)

    # Applying scaler
    df[STANDARD_SCALE_COLS] = scaler.transform(df[STANDARD_SCALE_COLS])
    df[LOG_SCALE_COLS] = np.log10(df[LOG_SCALE_COLS] + 1)
    return df


def create_new_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Creating new column using number of siblings(SibSp) and number of Parents/Children(Parch)
    df["Family_size"] = df["Parch"] + df["SibSp"]

    # Creating new column by encoding numeric tickets
    df["Numeric_ticket"] = df["Ticket"].apply(lambda x: 1 if x.isnumeric() else 0)

    # Creating new column by extracting name prefix (like Mr, Mrs, etc.)
    df["Name_title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    return df


def bin_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df["Family_size"] = pd.cut(df["Family_size"], bins=FAMILY_SIZE_BINS, labels=FAMILY_SIZE_LABELS)
    return df


def drop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns from df specified in COLS_TO_DROP
    :param df: pd.DataFrame
    :return: data frame
    """
    data = df.drop(columns=COLS_TO_DROP)
    return data


def preprocess_data(df: pd.DataFrame, split: bool = False, label: str = "Survived"):

    # Filling nan values using numerical and categorical imputers
    df = fill_na(df)

    # Creating additional data columns
    df = create_new_cols(df)

    # Binning FamilySize column
    df = bin_family_size(df)

    # Applying binary encoding and OneHotEncoder(to columns with diverse categories)
    df = encode(df)

    # Applying StandardScaler
    df = scale(df)

    # Dropping redundant columns
    df = drop(df)

    # Splitting DataFrame into numpy arrays of inputs and labels if split is True
    if split:
        labels = df[[label]].values.ravel()
        inputs = df.drop(columns=[label]).values.astype(np.float32)
        return inputs, labels

    inputs = df.values.astype(np.float32)
    return inputs
