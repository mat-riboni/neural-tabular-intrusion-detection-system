import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import numpy as np

logger = logging.getLogger(__name__)


def drop_nan_and_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    "Drop every row that contains NaN or +-Inf value"
    if not isinstance(df, pd.DataFrame):
        logger.error("df parameter is not a Pandas DataFrame")
        raise TypeError("df parameter must be a Pandas Dataframe")
    numerical_features = df.select_dtypes(include=np.number).columns.to_list()
    df_cleaned = df.copy()
    #replace all +- inf with NaN
    for col in numerical_features:
        df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
    #drop every row that contains NaN values
    df_cleaned.dropna(subset=numerical_features, inplace=True)
    removed = len(df) - len(df_cleaned)
    logger.info(f"{removed} rows dropped from DataFrame")
    return df_cleaned

def split_data(df: pd.DataFrame, random_state: int, target_column: str, target_category_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset in training set, validation set and test set"""
    X = df.drop(columns=[target_column ,target_category_column])
    y = df[target_column]
    stratify_data = df[target_column]
    # First split: training + validation  test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify_data
    )
    # if test set is 20%, train_val is l'80%. if val is 10% of the total, then is 10/80 = 12.5% of train_val
    relative_val_size = 0.125 # val_size / (1 - test_size) 
    stratify_train_val_data = y_train_val 
    # Second split: training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=stratify_train_val_data
    )
    logger.info(f"Shape: Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")
    # Recombine feature and target for pythorch
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df

