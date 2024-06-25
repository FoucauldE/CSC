import pandas as pd
from csc_lib.config import TRAIN_PATH, VAL_PATH, GEN_PATH

def load_data():
    df_train = pd.read_json(TRAIN_PATH)
    df_val = pd.read_json(VAL_PATH)
    df_gen = pd.read_json(GEN_PATH)
    return df_train, df_val, df_gen