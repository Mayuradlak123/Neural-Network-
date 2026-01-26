import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from config.logger import logger
from datetime import datetime
import json


class ETLFeatureService:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.transformations_history: List[Dict] = []

    def load_csv(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.original_df = self.df.copy()
        self.file_path = file_path
        return {"success": True}

    def handle_missing_values(self, strategy="median"):
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def export_data(self, output_path: str):
        self.df.to_csv(output_path, index=False)
        return {"success": True, "path": output_path}


# ===============================
# FEATURE ENGINEERING
# ===============================
def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering")
    X = X.copy()

    # Boolean mapping
    bool_map = {
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        True: 1, False: 0
    }

    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype(str).str.lower()
        X[col] = X[col].replace(bool_map)

    # Furnishing status ordinal encoding
    if "furnishingstatus" in X.columns:
        X["furnishingstatus"] = X["furnishingstatus"].map({
            "unfurnished": 0,
            "semi-furnished": 1,
            "furnished": 2
        }).fillna(0)

    # Log transform area
    if "area" in X.columns:
        X["log_area"] = np.log1p(X["area"])

    logger.info("Feature engineering completed")
    return X


# ===============================
# SESSION MANAGEMENT
# ===============================
_etl_sessions = {}

def get_etl_session(session_id: str) -> ETLFeatureService:
    if session_id not in _etl_sessions:
        _etl_sessions[session_id] = ETLFeatureService()
    return _etl_sessions[session_id]

def cleanup_etl_session(session_id: str):
    _etl_sessions.pop(session_id, None)
