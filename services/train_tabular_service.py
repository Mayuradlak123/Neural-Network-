import os
import time
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config.logger import logger


# -------------------------------------------------------
# ANN MODEL
# -------------------------------------------------------
class ANN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        logger.debug(f"Initializing ANN with input_dim={input_dim}, num_classes={num_classes}")
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


# -------------------------------------------------------
# ETL & CLEANING
# -------------------------------------------------------
def _detect_identifier_columns(df: pd.DataFrame) -> List[str]:
    """
    Heuristic to detect identifier columns.
    - column name contains 'id' (case-insensitive), or 'user' + 'id'
    - short unique values but numeric ids are still detected by name rule
    """
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
    """
    Load CSV and perform cleaning:
      - strip column names
      - drop user-requested columns
      - optionally drop identifier columns (like 'User ID')
      - remove duplicate rows
      - fill missing (numeric: median, categorical: mode)
    Returns: X (features DataFrame), y (Series)
    """
    logger.info(f"[ETL] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):
        logger.error(f"[ETL] CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"[ETL] Raw shape: {df.shape}")

    # normalize column names (strip whitespace only)
    df.columns = [c.strip() for c in df.columns]

    if label_col not in df.columns:
        logger.error(f"[ETL] Label column '{label_col}' not present in CSV columns: {df.columns.tolist()}")
        raise ValueError(f"Label column '{label_col}' not found")

    # optional user-specified drops
    if drop_columns:
        droplist = [col for col in drop_columns if col in df.columns]
        if droplist:
            logger.info(f"[ETL] Dropping user-specified columns: {droplist}")
            df = df.drop(columns=droplist)

    # auto-detect & drop identifier columns (do NOT drop label_col)
    if drop_identifier_columns:
        id_cols = _detect_identifier_columns(df)
        # ensure label isn't dropped accidentally (rare)
        id_cols = [c for c in id_cols if c != label_col]
        if id_cols:
            logger.info(f"[ETL] Dropping identifier columns: {id_cols}")
            df = df.drop(columns=id_cols)

    # remove exact duplicate rows (keep first)
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    removed = before - after
    if removed > 0:
        logger.info(f"[ETL] Removed {removed} duplicate rows")

    # split target / features
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # fill missing values
    logger.info("[ETL] Filling missing values (numeric -> median, categorical -> mode)")
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            if X[col].isnull().any():
                mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else ""
                X[col] = X[col].fillna(mode_val)
                logger.debug(f"[ETL] Filled categorical {col} with mode='{mode_val}'")
        else:
            if X[col].isnull().any():
                med = X[col].median()
                X[col] = X[col].fillna(med)
                logger.debug(f"[ETL] Filled numeric {col} with median={med}")

    logger.info(f"[ETL] Completed cleaning. Features shape: {X.shape}")
    return X, y


# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------
def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic feature engineering:
      - fills already applied in ETL are respected
      - create log_AnnualSalary, age_bin if columns exist
      - convert simple boolean-like strings to integers
    """
    logger.info("[FE] Starting feature engineering")
    X = X.copy()

    # numeric median fill as additional safety
    numeric_cols = X.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # categorical mode fill safety
    object_cols = X.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])

    # Feature transformations
    if "AnnualSalary" in X.columns:
        logger.debug("[FE] Creating log_AnnualSalary from AnnualSalary")
        X["AnnualSalary"] = X["AnnualSalary"].astype(float).clip(lower=0)
        X["log_AnnualSalary"] = np.log1p(X["AnnualSalary"])

    if "Age" in X.columns:
        logger.debug("[FE] Creating age_bin from Age")
        X["Age"] = X["Age"].astype(float).clip(lower=0)
        X["age_bin"] = pd.cut(X["Age"], bins=[-1, 29, 49, 200], labels=[0, 1, 2]).astype(int)

    # convert boolean-like strings to 0/1
    bool_map = {"True": 1, "False": 0, "true": 1, "false": 0, "Yes": 1, "No": 0, "yes": 1, "no": 0}
    for col in object_cols:
        # keep as object if truly categorical strings; but convert True/False/Yes/No
        X[col] = X[col].astype(str).str.strip()
        X[col] = X[col].replace(bool_map)

    logger.info(f"[FE] Feature engineering complete. Result shape: {X.shape}")
    return X


# -------------------------------------------------------
# BUILD TRANSFORMER
# -------------------------------------------------------
def build_transformer(X_df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """
    Build a ColumnTransformer pipeline:
      - OneHotEncode categorical columns (object dtype only)
      - StandardScale numeric columns
    Returns pipeline, numeric_cols, categorical_cols
    """
    logger.info("[TRANSFORMER] Building preprocessing pipeline")

    # Only treat object dtype columns as categorical (prevents numeric IDs being treated as categorical)
    categorical_cols = [c for c in X_df.columns if X_df[c].dtype == "object"]
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    logger.info(f"[TRANSFORMER] Numeric cols: {numeric_cols}")
    logger.info(f"[TRANSFORMER] Categorical cols: {categorical_cols}")

    # If there are no categorical or numeric columns, supply empty lists to avoid errors
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    pipeline = Pipeline([("transformer", transformer)])
    logger.info("[TRANSFORMER] Pipeline ready.")
    return pipeline, numeric_cols, categorical_cols


# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------
def train_tabular(
    csv_path: str,
    label_col: str,
    epochs: int = 50,
    lr: float = 0.001,
    drop_columns: Optional[List[str]] = None,
    drop_identifier_columns: bool = True,
    save_processed: bool = False,
    overwrite_raw: bool = False,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Full training pipeline.
    - save_processed: if True, writes processed CSV to processed/processed_<ts>.csv
    - overwrite_raw: if True AND save_processed==True, also overwrite the original csv_path
    """
    logger.info("======== Starting Tabular Training ========")

    # ETL
    X, y = etl_load_and_clean(csv_path, label_col, drop_columns, drop_identifier_columns)

    # Feature engineering
    X = feature_engineering(X)

    # save processed CSV if requested
    if save_processed:
        ts = int(time.time())
        processed_dir = "processed"
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, f"processed_{ts}.csv")
        processed_df = X.copy()
        processed_df[label_col] = y.values
        processed_df.to_csv(processed_path, index=False)
        logger.info(f"[SAVE] Processed dataset saved -> {processed_path}")

        if overwrite_raw:
            # overwrite original CSV (dangerous: do this only if you are sure)
            processed_df.to_csv(csv_path, index=False)
            logger.info(f"[SAVE] Original CSV overwritten with processed data -> {csv_path}")

    # Build transformer
    pipeline, num_cols, cat_cols = build_transformer(X)

    # Train/val split
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )

    # Fit transformer on training set only
    logger.info("[TRAIN] Fitting transformer on training data")
    pipeline.fit(X_train_df)

    # Transform datasets
    X_train = pipeline.transform(X_train_df)
    X_val = pipeline.transform(X_val_df)

    logger.info(f"[TRAIN] Transformed shapes — Train: {X_train.shape}, Val: {X_val.shape}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)

    input_dim = X_train.shape[1]
    num_classes = int(y.nunique())
    logger.info(f"[TRAIN] Model input_dim={input_dim}, num_classes={num_classes}")

    model = ANN(input_dim=input_dim, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}

    logger.info(f"[TRAIN] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        history["train_loss"].append(float(loss.item()))
        history["val_acc"].append(float(val_acc))

        logger.info(f"[EPOCH {epoch}/{epochs}] train_loss={loss.item():.4f} val_acc={val_acc:.4f}")

    # Save model + transformer
    ts = int(time.time())
    os.makedirs("models/tabular", exist_ok=True)
    model_path = f"models/tabular/tabular_model_{ts}.pt"
    transformer_path = f"models/tabular/transformer_{ts}.pkl"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "transformer_path": transformer_path,
    }

    torch.save(checkpoint, model_path)
    joblib.dump(pipeline, transformer_path)

    logger.info(f"[SAVE] Model saved -> {model_path}")
    logger.info(f"[SAVE] Transformer saved -> {transformer_path}")
    logger.info("======== Training Complete ========")

    return {
        "success": True,
        "model_path": model_path,
        "transformer_path": transformer_path,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "history": history,
    }


# -------------------------------------------------------
# PREDICT
# -------------------------------------------------------
def predict_tabular(model_path: str, data: dict) -> Dict[str, Any]:
    """
    Load model checkpoint (which contains transformer path) and run inference on single row dict.
    """
    logger.info(f"[PREDICT] Loading model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"[PREDICT] Model not found: {model_path}")
        raise FileNotFoundError(model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    transformer_path = checkpoint.get("transformer_path")
    if transformer_path is None or not os.path.exists(transformer_path):
        logger.error(f"[PREDICT] Transformer not found at: {transformer_path}")
        raise FileNotFoundError(transformer_path)

    pipeline = joblib.load(transformer_path)
    model = ANN(input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"[PREDICT] Incoming data: {data}")

    # Convert incoming dict to DataFrame and apply same FE
    df = pd.DataFrame([data])
    df = feature_engineering(df)

    # transform & infer
    X = pipeline.transform(df)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_class = int(pred.item())
    confidence = float(conf.item())
    logger.info(f"[PREDICT] Prediction Class={pred_class}, Confidence={confidence:.4f}")

    return {"pred_class": pred_class, "confidence": confidence}
