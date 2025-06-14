import joblib
import pandas as pd
from config import ENV_FILE_PATH
from dotenv import dotenv_values
from preprocessing import preprocess_data


def common_encoder() -> None:
    """
    Runs trained model on test data and saved results to submission file
    :return: None
    """

    # Initializing dataframe
    env_config = dotenv_values(ENV_FILE_PATH)
    df_test = pd.read_csv(env_config["TEST_DATASET_PATH"])
    ids = df_test["PassengerId"].tolist()

    # Preprocessing dataframe
    test_data = preprocess_data(df_test)

    # Loading model from the dump
    model = joblib.load(env_config["MODEL_DUMP_PATH"])

    # Making predictions on test data
    y_pred = model.predict(test_data)

    # Saving results to submission file
    submission_data = {"PassengerId": ids, "Survived": y_pred}
    df = pd.DataFrame(submission_data)
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    common_encoder()
