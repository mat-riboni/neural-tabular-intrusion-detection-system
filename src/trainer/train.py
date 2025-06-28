# train.py
import torch
import argparse
import logging
import os
import joblib

from src.utilities.config_manager import ConfigManager
from src.utilities.io_handler import load_data
from src.utilities.dataset_utils import drop_nan_and_inf_values, split_data
from src.data.preprocessor import TabnetPreprocessor
from pytorch_tabnet.tab_model import TabNetClassifier
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    """Main function for model training"""

    DATASET = args.dataset

    if DATASET == 'ton':
        DATASET_CONFIG_PATH = './config/ton_config.json'
    else:
        DATASET_CONFIG_PATH = './config/netflow_config.json'
    
    TABNET_CONFIG_PATH = './config/tabnet_config.json'

    try:
        ConfigManager.load_config(DATASET_CONFIG_PATH)
        logger.info(f"Dataset configuration loaded successfully from {DATASET_CONFIG_PATH}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error during dataset configuration loading: {e}")
        return
    paths_config = ConfigManager.get_section("paths")
    data_cols_config = ConfigManager.get_section("data_columns")
  

    DATA_PATH = paths_config.get("dataset_path")
    OUTPUT_DIR = paths_config.get("output_dir")
    TARGET_BINARY_COL = data_cols_config.get("target_column")
    TARGET_MULTICLASS_COL = data_cols_config.get("target_category_column")
    NUMERICAL_COLS = data_cols_config.get("numerical_cols")
    CATEGORICAL_COLS = data_cols_config.get("categorical_cols")

    if args.classification == 'multiclass':
        TARGET_COL = TARGET_MULTICLASS_COL
    else:
        TARGET_COL = TARGET_BINARY_COL

    try:
        ConfigManager.load_config(TABNET_CONFIG_PATH)
        logger.info(f"TabNet configuration loaded successfully from {TABNET_CONFIG_PATH}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error during TabNet configuration loading: {e}")
        return
    training_params = ConfigManager.get_section("training_settings")
    all_hyperparams = ConfigManager.get_section("hyperparameters")
    RANDOM_STATE = training_params.get("random_state", 42)
    
    try:
        params = all_hyperparams[args.model_size]
        logger.info(f"Hyperparameters loaded '{args.model_size}'.")
    except KeyError:
        logger.error(f"Model size '{args.model_size}' not found.")
        return


    df = load_data(DATA_PATH)

    train_df, val_df, test_df = split_data(df, random_state=RANDOM_STATE, target_binary_column=TARGET_BINARY_COL, target_multiclass_column=TARGET_MULTICLASS_COL, target_column=TARGET_COL)
    
    train_df = drop_nan_and_inf_values(train_df)
    val_df = drop_nan_and_inf_values(val_df)
    test_df = drop_nan_and_inf_values(test_df)

    test_df.to_csv(f'./resources/dataset/test_set_{DATASET}.csv', index=False) #Save test data

    y_train = train_df[TARGET_COL]
    X_train = train_df.drop(columns=[TARGET_COL]) 

    y_val = val_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])

    if args.classification == 'multiclass':
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)   
        y_val_enc   = le.transform(y_val)         

        encoder_path = os.path.join(
            OUTPUT_DIR,
            f'label_encoder_{args.classification}_{args.model_size}.pkl'
        )
        joblib.dump(le, encoder_path)
        logger.info(f"LabelEncoder saved in: {encoder_path}")

    else:  # binary
        y_train_enc = y_train.values #needed numpy array for tabnet
        y_val_enc   = y_val.values   #needed numpy array for tabnet

    logger.info(f"Dimensions: Train={X_train.shape}, Validation={X_val.shape}, Test={test_df.shape}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    if args.use_weights:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
        weights_tensor = torch.tensor(cw, dtype=torch.float).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        logger.info(f"Using automatically balanced class weights: {cw}")
    else:
        loss_fn = nn.CrossEntropyLoss()
        logger.info("Using unweighted loss (no class weights)")


   
    preprocessor = TabnetPreprocessor(numerical_cols=NUMERICAL_COLS, categorical_cols=CATEGORICAL_COLS)
    logger.info("TabnetPreprocessor fitting...")
    preprocessor.fit(X_train)
    
    logger.info("Data tranformation...")
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    cat_dims, cat_idxs = preprocessor.get_tabnet_params()
    

    logger.info("TabNet configuration...")

    clf = TabNetClassifier(
        n_d=params['n_d'], n_a=params['n_a'], n_steps=params['n_steps'],
        gamma=params['gamma'], cat_idxs=cat_idxs, cat_dims=cat_dims,
        cat_emb_dim=params['cat_emb_dim'], lambda_sparse=params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params['lr'], weight_decay=params['weight_decay']),
        mask_type='sparsemax', device_name=device, seed=RANDOM_STATE
    )

    logger.info(f"Starting {args.classification} {args.model_size} model training with {DATASET} dataset...")


    clf.fit(
        X_train=X_train_processed.values, y_train=y_train_enc,
        eval_set=[(X_val_processed.values, y_val_enc)],
        eval_name=['validation'], eval_metric=['accuracy', 'balanced_accuracy'],
        max_epochs=params['max_epochs'],
        patience=training_params.get("early_stopping_patience", 20),
        batch_size=params['batch_size'],
        virtual_batch_size=training_params.get("virtual_batch_size", 128),
        loss_fn=loss_fn
    )
    logger.info("Training complete.")
    
    #save the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(OUTPUT_DIR, f'tabnet_{args.classification}_{args.model_size}_{DATASET}')
    clf.save_model(model_path)
    logger.info(f"Model saved in: {model_path}.zip")

    preprocessor_path = os.path.join(OUTPUT_DIR, f'preprocessor_{args.classification}_{args.model_size}_{DATASET}.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved in: {preprocessor_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TabNet training script.")
    parser.add_argument(
        '--model-size', 
        type=str, 
        default='small', 
        choices=['small', 'medium', 'large'], 
        help="Model dimensions, see config.json"
    )
    parser.add_argument(
        '--classification', 
        type=str, 
        default='binary', 
        choices=['binary', 'multiclass'], 
        help="Classification type: binary or multiclass"
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='ton', 
        choices=['ton', 'netflow'], 
        help="Dataset used"
    )
    parser.add_argument(
        '--use-weights',
        action='store_true',
        help='If set, uses balanced class weights in the loss function'
    )

    args = parser.parse_args()
    main(args)