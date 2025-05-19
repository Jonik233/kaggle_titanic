# import torch
import joblib
import pandas as pd
from config import ENV_FILE_PATH
from dotenv import dotenv_values
from preprocessing import preprocess_data

# from models import Model
from transforms import TransformerPipeline
from train import train

# def network_encoder():
#     df_test = pd.read_csv("data/test.csv")
#     ids = df_test["PassengerId"].tolist()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     pipeline = TransformerPipeline()
#     test_data = pipeline.transform(df_test)
#     test_data = torch.from_numpy(test_data).to(device=device)
#
#     with torch.no_grad():
#         model = Model(10).to(device=device)
#         state_dict = torch.load("weights.pth")
#         model.load_state_dict(state_dict)
#
#         model.eval()
#         probs = model(test_data)
#         y_pred = torch.where(probs > 0.5, 1, 0)
#
#         submission_data = {"PassengerId":ids, "Survived":y_pred.cpu()}
#         df = pd.DataFrame(submission_data)
#         df.to_csv("submission.csv", index=False)


def common_encoder():
    env_config = dotenv_values(ENV_FILE_PATH)
    df_test = pd.read_csv(env_config["TEST_DATASET_PATH"])
    ids = df_test["PassengerId"].tolist()

    test_data = preprocess_data(df_test)
    model = joblib.load(env_config["MODEL_DUMP_PATH"])

    y_pred = model.predict(test_data)

    submission_data = {"PassengerId": ids, "Survived": y_pred}
    df = pd.DataFrame(submission_data)
    df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    common_encoder()
