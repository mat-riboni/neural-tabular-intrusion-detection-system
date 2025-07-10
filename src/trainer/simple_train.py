import torch
import logging
import os
import joblib

from src.utilities.config_manager import ConfigManager
from src.utilities.io_handler import load_data
from src.utilities.dataset_utils import *
from pytorch_tabnet.tab_model import TabNetClassifier
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_CONFIG_PATH = './config/ton_config.json'
TABNET_CONFIG_PATH = './config/tabnet_config.json'



def main():


    ConfigManager.load_config(DATASET_CONFIG_PATH)
    logger.info(f"Dataset configuration loaded successfully from {DATASET_CONFIG_PATH}")
    
    paths_config = ConfigManager.get_section("paths")
    data_cols_config = ConfigManager.get_section("data_columns")
  

    DATA_PATH = paths_config.get("dataset_path")
    OUTPUT_DIR = paths_config.get("output_dir")
    TARGET_COL = data_cols_config.get("target_category_column")
    NUMERICAL_COLS = data_cols_config.get("numerical_cols")
    CATEGORICAL_COLS = data_cols_config.get("categorical_cols")
    RANDOM_STATE = 42    

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)

    keep_cols = CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL]
    df = df[keep_cols].copy() 

    #Data split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df[TARGET_COL])   
    test_df.to_csv(f'./resources/dataset/test_set_ton.csv', index=False)

    logger.info(f"Encoding...")

    scaler = StandardScaler()
    scaler.fit(train_df[NUMERICAL_COLS])
    for _df in (train_df, valid_df, test_df):
        _df[NUMERICAL_COLS] = scaler.transform(_df[NUMERICAL_COLS])
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "simple_num_scaler.pkl"))


    categorical_dims, encoders = {}, {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder().fit(train_df[col])
        le.classes_ = np.append(le.classes_, "_UNK")
        train_df[col] = le.transform(train_df[col])
        valid_df[col] = le.transform(
            valid_df[col].where(valid_df[col].isin(le.classes_), "_UNK")
        )
        test_df[col] = le.transform(
            test_df[col].where(test_df[col].isin(le.classes_), "_UNK")
        )
        categorical_dims[col] = len(le.classes_)   
        encoders[col] = le                         


    y_le = LabelEncoder().fit(train_df[TARGET_COL])
    for _df in (train_df, valid_df, test_df):
        _df[TARGET_COL] = y_le.transform(_df[TARGET_COL])

    unused_feat = [ col for col in df.columns if col not in NUMERICAL_COLS + CATEGORICAL_COLS]

    features = [ col for col in df.columns if col not in unused_feat+[TARGET_COL]] 

    cat_idxs = [ i for i, f in enumerate(features) if f in CATEGORICAL_COLS]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in CATEGORICAL_COLS]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=4,
        gamma=1.8, n_independent=2, n_shared=2,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=[min(50, (dim + 1) // 2) for dim in cat_dims], 
        lambda_sparse=1e-2,
        momentum=0.02,
        clip_value=1.5,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),  
        scheduler_params={"gamma": 0.95, "step_size": 20},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        epsilon=1e-15, device_name=device
    )


    X_train = train_df[features].values
    y_train = train_df[TARGET_COL].values

    X_valid = valid_df[features].values
    y_valid = valid_df[TARGET_COL].values


    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = cw / cw.mean() # to avoid huge differences
    weights_tensor = torch.tensor(cw, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    logger.info(f"Using automatically balanced class weights: {cw}")

    logger.info("Starting training")


    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        max_epochs=100, patience=20,
        batch_size=2048, virtual_batch_size=256,
        loss_fn=loss_fn,
        eval_metric= ['balanced_accuracy', 'accuracy']
    )

    
    model_path = os.path.join(OUTPUT_DIR, 'tabnet_simple')
    clf.save_model(model_path)
    logger.info(f"Model saved in: {model_path}.zip")

    encoder_path = os.path.join(
            OUTPUT_DIR,
            f'simple_label_encoder.pkl'
        )
    joblib.dump(encoders, encoder_path)
    joblib.dump(y_le, os.path.join(OUTPUT_DIR, "simple_target_encoder.pkl"))

    logger.info(f"LabelEncoder saved in: {encoder_path}")

if __name__ == '__main__':
    main()