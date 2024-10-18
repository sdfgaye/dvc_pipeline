from logger import logging
from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(config):
    logging.info("Preparing data")
    df = pd.read_csv(config.data.csv_file_path)
    logging.info("Raw data imported")
    df['labels'] = pd.factorize(df['sentiment'])[0]

    test_size = config.data.test_set_ratio
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=123)

    logging.info("data splitted")

    train_df.to_csv(config.data.train_csv_save_path, index = False)
    test_df.to_csv(config.data.test_csv_save_path, index = False)

    logging.info("split data saved")

if __name__ == "__main__":
    config = OmegaConf.load("./params.yaml")
    prepare_data(config)