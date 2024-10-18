from sklearn.linear_model import LogisticRegression
from omegaconf import OmegaConf
from logger import logging
import joblib
import pandas as pd



def train(config):
    logging.info("preparing traing")
    train_inputs = joblib.load(config.features.train_features_save_path)
    train_outputs = pd.read_csv(config.data.train_csv_save_path)['labels'].values

    logging.info("model initiation")
    penalty = config.train.penalty
    C = config.train.C
    solver = config.train.solver

    model = LogisticRegression(penalty=penalty, C=C, solver=solver)
    model.fit(train_inputs, train_outputs)
    logging.info("model trained")

    joblib.dump(model, config.train.model_save_path)
    logging.info("model dumped")


if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    train(config)

