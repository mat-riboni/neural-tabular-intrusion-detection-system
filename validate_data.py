import pandas as pd
import logging
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# Importa i tuoi moduli
from src.utilities.config_manager import ConfigManager
from src.utilities.io_handler import load_data
from src.utilities.dataset_utils import drop_nan_and_inf_values, split_data
from src.data.preprocessor import TabnetPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

CONFIG_PATH = './config/config.json'

def validate_data_leakage():
    """
    Trains a simple Decision Tree to check for obvious data leaks.
    """
    logging.info("--- Inizio Test di Data Leakage con Albero Decisionale ---")

    # Carica la configurazione e i dati esattamente come in train.py
    ConfigManager.load_config(CONFIG_PATH)
    paths_config = ConfigManager.get_section("paths")
    data_cols_config = ConfigManager.get_section("data_columns")
    training_params = ConfigManager.get_section("training_settings")

    numerical_cols = data_cols_config.get("numerical_cols", [])
    categorical_cols = data_cols_config.get("categorical_cols", [])
    target_col = data_cols_config.get("target_column")
    target_category_col = data_cols_config.get("target_category_column")
    
    all_needed_cols = list(set(numerical_cols + categorical_cols + [target_col, target_category_col]))

    logging.info(f"Loading dataset from {paths_config.get('dataset_path')}...")
    df = pd.read_csv(paths_config.get("dataset_path"), usecols=all_needed_cols)
    df_cleaned = drop_nan_and_inf_values(df)

    train_df, val_df, _ = split_data(df_cleaned, 
                                     random_state=training_params.get("random_state", 42), 
                                     target_column=target_col, 
                                     target_category_column=target_category_col)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    # Usa lo stesso preprocessore
    preprocessor = TabnetPreprocessor(numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    logging.info("Dati processati. Addestramento dell'Albero Decisionale...")

    # Addestra un albero decisionale molto semplice (profondità massima = 2)
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=training_params.get("random_state", 42))
    tree_clf.fit(X_train_processed, y_train)

    # Valuta sul set di validazione
    y_pred = tree_clf.predict(X_val_processed)
    accuracy = accuracy_score(y_val, y_pred)

    logging.info(f"Accuracy dell'Albero Decisionale (max_depth=2): {accuracy:.6f}")

    if accuracy > 0.95:
        logging.warning("!!! L'accuracy è estremamente alta. Questo conferma un data leak. !!!")
        
        # Esporta e stampa la regola che l'albero ha imparato
        feature_names = X_train_processed.columns.tolist()
        tree_rules = export_text(tree_clf, feature_names=feature_names)
        
        print("\n" + "="*20 + " REGOLA DI LEAK TROVATA " + "="*20)
        print("L'albero decisionale ha trovato questa semplice regola per separare i dati:")
        print(tree_rules)
        print("="*64)
    else:
        logging.info("L'accuracy è in un range realistico. Il leak potrebbe non essere così ovvio.")
        
    print("\nReport di Classificazione Completo:")
    print(classification_report(y_val, y_pred))


if __name__ == '__main__':
    validate_data_leakage()