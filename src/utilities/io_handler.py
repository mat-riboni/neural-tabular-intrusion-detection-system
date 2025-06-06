import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    logger.info(f"Loading path: {file_path}")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found in {file_path}")