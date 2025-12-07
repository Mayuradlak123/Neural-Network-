# services/train_regression_service.py

import os
import time
import joblib
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config.logger import logger
from services.train_tabular_service import etl_load_and_clean, feature_engineering


# ------------------------------------
# Regression ANN Model
# ------------------------------------
class ANNRegressor(nn.Module):
    def __init__(self, input_dim: int):
        logger.debug(f"Initializing ANNRegressor with input_dim={input_dim}")
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


# ------------------------------------
# Build Transformer
# ------------------------------------
def build_transformer_regression(X_df: pd.DataFrame):
    logger.info("Building preprocessing transformer for regression...")

    categorical_cols = [
        c for c in X_df.columns
        if X_df[c].dtype == "object" or X_df[c].nunique() < 15
    ]
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    logger.debug(f"Identified categorical columns: {categorical_cols}")
    logger.debug(f"Identified numeric columns: {numeric_cols}")

    pipeline = Pipeline([
        (
            "transformer",
            ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
                ],
                remainder="drop",
                sparse_threshold=0
            )
        )
    ])

    logger.info("Transformer pipeline created.")

    return pipeline, numeric_cols, categorical_cols


# ------------------------------------
# TRAIN REGRESSION MODEL
# ------------------------------------
def train_regression(csv_path: str, label_col: str, epochs: int = 50, lr: float = 0.001):
    logger.info("======== Starting Regression Training ========")
    logger.info(f"Loading CSV: {csv_path} | label column: {label_col}")

    X, y = etl_load_and_clean(csv_path, label_col)
    logger.debug(f"Initial dataset shape: X={X.shape}, y={y.shape}")

    logger.info("Running feature engineering...")
    X = feature_engineering(X)
    logger.debug(f"Dataset shape after feature engineering: {X.shape}")

    logger.info("Building transformer...")
    pipeline, _, _ = build_transformer_regression(X)

    logger.info("Splitting dataset (80/20)...")
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.debug(f"Train shapes: X={X_train_df.shape}, y={y_train.shape}")
    logger.debug(f"Val shapes: X={X_val_df.shape}, y={y_val.shape}")

    logger.info("Fitting preprocessing pipeline to training data...")
    pipeline.fit(X_train_df)

    logger.info("Transforming train and validation sets...")
    X_train = pipeline.transform(X_train_df)
    X_val = pipeline.transform(X_val_df)

    logger.debug(f"Transformed X_train shape: {X_train.shape}")
    logger.debug(f"Transformed X_val shape: {X_val.shape}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    logger.info("Initializing ANN Regression model...")
    input_dim = X_train.shape[1]
    model = ANNRegressor(input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Starting training loop...")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))

        logger.info(f"[EPOCH {epoch+1}/{epochs}] Loss={loss:.4f}, ValLoss={val_loss:.4f}")

    logger.info("Training completed.")

    ts = int(time.time())
    os.makedirs("models/regression", exist_ok=True)

    model_path = f"models/regression/regression_model_{ts}.pt"
    transformer_path = f"models/regression/transformer_{ts}.pkl"

    logger.info(f"Saving model to {model_path}")
    logger.info(f"Saving transformer to {transformer_path}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "transformer_path": transformer_path,
    }

    torch.save(checkpoint, model_path)
    joblib.dump(pipeline, transformer_path)

    logger.info("Model and transformer saved successfully.")

    return {
        "success": True,
        "model_path": model_path,
        "transformer_path": transformer_path,
        "input_dim": input_dim,
        "history": history
    }


# ------------------------------------
# PREDICT REGRESSION
# ------------------------------------
def predict_regression(model_path: str, data: dict) -> Dict[str, Any]:
    logger.info(f"Loading model for prediction: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    transformer_path = checkpoint["transformer_path"]

    logger.info(f"Loading transformer: {transformer_path}")
    pipeline = joblib.load(transformer_path)

    logger.info("Preparing input data for prediction...")
    df = pd.DataFrame([data])
    logger.debug(f"Raw input data:\n{df}")

    df = feature_engineering(df)
    logger.debug(f"Data after feature engineering:\n{df}")

    X = pipeline.transform(df)
    logger.debug(f"Transformed input shape: {X.shape}")

    X_t = torch.tensor(X, dtype=torch.float32)

    logger.info("Loading ANN model weights...")
    model = ANNRegressor(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Running prediction...")
    with torch.no_grad():
        pred = model(X_t).item()

    logger.info(f"Prediction result: {pred}")

    return {"prediction": pred}
