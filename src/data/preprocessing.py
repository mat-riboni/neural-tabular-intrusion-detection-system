import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Aggiungi altro
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    logger.info(f"Loading path: {file_path}")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found in {file_path}")




def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset in training set, validation set and test set"""
    
    target_column = 'Label'

    X = df.drop(columns=[target_column ,'Attack'])
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

