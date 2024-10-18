import pandas as pd
import joblib
import pandas as pd
from omegaconf import OmegaConf
from logger import logging
from sklearn.metrics import accuracy_score, f1_score


def evaluate(config):
    logging.info("Evalution started")
    test_inputs = joblib.load(config.features.test_features_save_path)
    test_df = pd.read_csv(config.data.test_csv_save_path)
    logging.info("tests info loaded")

    test_outputs = test_df["labels"].values
    class_names = test_df["sentiment"].unique().tolist()

    model = joblib.load(config.train.model_save_path)
    logging.info("loading trained model")

    
    
    metric_name = config.evaluate.metric
    metric = {
        "accuracy": accuracy_score,
        "f1_score": f1_score
    }[metric_name]

    logging.info("predicting")
    predicted_test_outputs = model.predict(test_inputs)
    
    logging.info("calculating evaluation metrics")
    result = metric(test_outputs, predicted_test_outputs)
    result_dict = {metric_name: float(result)}

    OmegaConf.save(result_dict, config.evaluate.results_save_path)
    logging.info("Metrics saved")




if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    evaluate(config)