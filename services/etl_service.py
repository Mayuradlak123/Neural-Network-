# services/etl_service.py

import os
import pandas as pd
from typing import Optional, List, Tuple
from config.logger import logger


def _detect_identifier_columns(df: pd.DataFrame) -> List[str]:
    id_cols = []
    for c in df.columns:
        cname = c.strip().lower()
        if "id" in cname or cname in ("userid", "user id", "id"):
            id_cols.append(c)
    return id_cols


def etl_load_and_clean(
    csv_path: str,
    label_col: str,
    drop_columns: Optional[List[str]] = None,
    drop_identifier_columns: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:

    logger.info(f"[ETL] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"[ETL] Raw shape: {df.shape}")

    df.columns = [c.strip() for c in df.columns]

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])

    if drop_identifier_columns:
        id_cols = _detect_identifier_columns(df)
        id_cols = [c for c in id_cols if c != label_col]
        if id_cols:
            df = df.drop(columns=id_cols)

    df = df.drop_duplicates()

    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == "object":
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])
        else:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

    logger.info("[ETL] Completed cleaning")
    return X, y
