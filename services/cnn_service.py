import os
import time
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import uuid
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from services.etl_feature_service import (
    get_etl_session,
    cleanup_etl_session,
    feature_engineering
)
from config.logger import logger


# ===============================
# 1D CNN for Tabular Regression
# ===============================
class TabularCNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        pooled_dim = input_dim // 2
        self.fc1 = nn.Linear(32 * pooled_dim, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ===============================
# TRAINING PIPELINE
# ===============================
def train_cnn_tabular(
    file_path: str,
    label_col: str,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32
) -> Dict[str, Any]:

    session_id = str(uuid.uuid4())

    try:
        logger.info("Starting CNN training pipeline")
        etl = get_etl_session(session_id)

        # ---------------------------
        # ETL
        # ---------------------------
        etl.load_csv(file_path)
        etl.handle_missing_values(strategy="median")

        df = etl.df

        # Feature Engineering
        df = feature_engineering(df)

        # Save processed CSV
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        processed_csv_path = os.path.join(
            processed_dir, f"processed_{int(time.time())}.csv"
        )
        df.to_csv(processed_csv_path, index=False)

        # ---------------------------
        # Encoding & Scaling
        # ---------------------------
        cat_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and c != label_col
        ]
        num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != label_col
        ]

        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        feature_scaler = StandardScaler()
        df[num_cols] = feature_scaler.fit_transform(df[num_cols])

        target_scaler = StandardScaler()
        df[[label_col]] = target_scaler.fit_transform(df[[label_col]])

        X = df.drop(columns=[label_col])
        y = df[label_col]

        # ---------------------------
        # Train / Val Split
        # ---------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        # ---------------------------
        # Model
        # ---------------------------
        model = TabularCNN(input_dim=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
                )

        # ---------------------------
        # Save Artifacts
        # ---------------------------
        ts = int(time.time())
        model_dir = "models/cnn"
        os.makedirs(model_dir, exist_ok=True)

        model_path = f"{model_dir}/cnn_tabular_{ts}.pt"
        metadata_path = f"{model_dir}/cnn_metadata_{ts}.joblib"
        background_path = f"{model_dir}/cnn_background_{ts}.joblib"

        torch.save(model.state_dict(), model_path)
        joblib.dump(X_train_t[:50], background_path)

        metadata = {
            "input_dim": X.shape[1],
            "columns": list(X.columns),
            "label_col": label_col,
            "encoders": encoders,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "background_path": background_path
        }

        joblib.dump(metadata, metadata_path)

        return {
            "success": True,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "processed_csv_path": processed_csv_path,
            "history": history,
            "session_id": session_id
        }

    finally:
        cleanup_etl_session(session_id)


# ===============================
# PREDICTION + LOSS + ACTIONS
# ===============================
def predict_cnn_tabular(
    model_path: str,
    metadata_path: str,
    data: Dict[str, Any]
) -> Dict[str, Any]:

    metadata = joblib.load(metadata_path)
    model = TabularCNN(metadata["input_dim"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    df = pd.DataFrame([data])
    df = feature_engineering(df)

    # Encoding
    for col, le in metadata["encoders"].items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Scaling
    df[metadata["num_cols"]] = metadata["feature_scaler"].transform(
        df[metadata["num_cols"]]
    )

    for col in metadata["columns"]:
        if col not in df.columns:
            df[col] = 0

    X = df[metadata["columns"]]
    X_t = torch.tensor(X.values, dtype=torch.float32)

    criterion = nn.MSELoss()

    with torch.no_grad():
        pred_scaled = model(X_t)
        pred = metadata["target_scaler"].inverse_transform(
            pred_scaled.numpy()
        )[0][0]

    # Optional loss
    loss_val = None
    if "actual_price" in data:
        actual = np.array([[data["actual_price"]]])
        actual_scaled = metadata["target_scaler"].transform(actual)
        loss_val = criterion(
            torch.tensor(actual_scaled, dtype=torch.float32),
            pred_scaled
        ).item()

    # Actionability
    actions = []
    if data.get("airconditioning") in ["no", 0]:
        actions.append("Adding air conditioning can increase price by 5–10%")

    if data.get("furnishingstatus", "") != "furnished":
        actions.append("Fully furnished homes have higher resale value")

    if data.get("parking", 0) < 1:
        actions.append("Adding parking increases demand significantly")

    return {
        "prediction_price": float(pred),
        "loss_mse": loss_val,
        "loss_function": "MSE",
        "recommendations": actions
    }
