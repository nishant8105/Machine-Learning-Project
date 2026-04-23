import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')

def load_data(path : str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File Not Found at {path}\n"
            "Download from : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud \n"
            "Place creditcard.csv data/raw/"
            )
    df = pd.read_csv(path)
    return df

def get_summary(dataframe : pd.DataFrame) -> dict:
    shape = dataframe.shape
    total = len(dataframe)
    fraud_count = dataframe['Class'].sum()
    legit = total - fraud_count

    return{
        "Shape" : shape,
        "Total_transactions": total,
        "fraud_cases" : int(fraud_count),
        "legit_cases" : int(legit),
        "fraud_percent" : round(fraud_count/total * 100, 4),
        "missing_value": int(dataframe.isnull().sum().sum()),
        "features" : dataframe.shape[1] - 1,
        "memory_mb" : round(dataframe.memory_usage(deep = True).sum()/1e6, 2)
    }

def get_feature_target(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y