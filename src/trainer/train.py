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

CONFIG_PATH = './config/short_config.json'

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__)

def main(args):
    """Main function for model training"""
    
    try:
        ConfigManager.load_config(CONFIG_PATH)
        logging.info(f"Configuration loaded successfully from {CONFIG_PATH}")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error during configuration loading: {e}")
        return

    #Get all necessary configurations
    paths_config = ConfigManager.get_section("paths")
    data_cols_config = ConfigManager.get_section("data_columns")
    training_params = ConfigManager.get_section("training_settings")
    all_hyperparams = ConfigManager.get_section("hyperparameters")

    DATA_PATH = paths_config.get("dataset_path")
    OUTPUT_DIR = paths_config.get("output_dir")
    TARGET_COL = data_cols_config.get("target_column")
    TARGET_CATEGORY_COL = data_cols_config.get("target_category_column")
    NUMERICAL_COLS = data_cols_config.get("numerical_cols")
    CATEGORICAL_COLS = data_cols_config.get("categorical_cols")
    RANDOM_STATE = training_params.get("random_state", 42)
    
    try:
        params = all_hyperparams[args.model_size]
        logging.info(f"Hyperparameters loaded '{args.model_size}'.")
    except KeyError:
        logging.error(f"Model size '{args.model_size}' not found.")
        return

    df = load_data(DATA_PATH)

    train_df, val_df, test_df = split_data(df, random_state=RANDOM_STATE, target_column=TARGET_COL, target_category_column=TARGET_CATEGORY_COL)
    
    train_df = drop_nan_and_inf_values(train_df)
    val_df = drop_nan_and_inf_values(val_df)
    test_df = drop_nan_and_inf_values(test_df)


    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]
    
   

    logging.info(f"Dimensions: Train={X_train.shape}, Validation={X_val.shape}, Test={test_df.shape}")

    # Pre-processing
    preprocessor = TabnetPreprocessor(numerical_cols=NUMERICAL_COLS, categorical_cols=CATEGORICAL_COLS)
    logging.info("TabnetPreprocessor fitting...")
    preprocessor.fit(X_train)
    
    logging.info("Data tranformation...")
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    cat_dims, cat_idxs = preprocessor.get_tabnet_params()
    

    logging.info("TabNet configuration...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device:: {device}")

    clf = TabNetClassifier(
        n_d=params['n_d'], n_a=params['n_a'], n_steps=params['n_steps'],
        gamma=params['gamma'], cat_idxs=cat_idxs, cat_dims=cat_dims,
        cat_emb_dim=params['cat_emb_dim'], lambda_sparse=params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params['lr'], weight_decay=params['weight_decay']),
        mask_type='sparsemax', device_name=device, seed=RANDOM_STATE
    )

    logging.info("Starting model training...")
    clf.fit(
        X_train=X_train_processed.values, y_train=y_train.values,
        eval_set=[(X_val_processed.values, y_val.values)],
        eval_name=['validation'], eval_metric=['auc', 'accuracy', 'balanced_accuracy'],
        max_epochs=params['max_epochs'],
        patience=training_params.get("early_stopping_patience", 20),
        batch_size=params['batch_size'],
        virtual_batch_size=training_params.get("virtual_batch_size", 128)
    )
    logging.info("Training complete.")
    
    #save the model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(OUTPUT_DIR, f'tabnet_model_{args.model_size}')
    clf.save_model(model_path)
    logging.info(f"Model saved in: {model_path}.zip")

    preprocessor_path = os.path.join(OUTPUT_DIR, f'preprocessor_{args.model_size}.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logging.info(f"Preprocessor saved in: {preprocessor_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TabNet training script.")
    parser.add_argument(
        '--model-size', 
        type=str, 
        default='small', 
        choices=['small', 'medium', 'large'], 
        help="Model dimensions, see config.json"
    )
    args = parser.parse_args()
    main(args)