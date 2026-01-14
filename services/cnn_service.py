import os
import time
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import uuid
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from services.etl_feature_service import get_etl_session, cleanup_etl_session
from services.model_interpretability_service import InterpretabilityService
from config.logger import logger

# 1D CNN for Tabular Data (Regression)
class TabularCNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Input shape: (Batch, 1, input_dim)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Calculate dimension after pooling
        pooled_dim = input_dim // 2
        
        self.fc1 = nn.Linear(32 * pooled_dim, 64)
        self.fc2 = nn.Linear(64, 1) # Regression output
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is (Batch, input_dim) -> reshape to (Batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cnn_tabular(
    file_path: str,
    label_col: str,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    save_processed: bool = True
) -> Dict[str, Any]:
    """
    CNN Training Pipeline for Tabular Data
    """
    session_id = str(uuid.uuid4())
    try:
        logger.info(f"Starting CNN training for {file_path}")
        etl_service = get_etl_session(session_id)
        
        # 1. ETL & Feature Engineering
        etl_service.load_csv(file_path)
        
        # Generic cleaning
        etl_service.handle_missing_values(strategy='median')
        
        # Identify categorical and numeric
        df = etl_service.df
        # Everything not strictly numeric is treated as categorical for encoding
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != label_col]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != label_col]
        
        logger.info(f"Detected categorical columns: {cat_cols}")
        logger.info(f"Detected numeric columns: {num_cols}")
        
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        scaler = None
        if num_cols:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            
        # Get processed data
        processed_df = df
        X = processed_df.drop(columns=[label_col])
        y = processed_df[label_col]
        
        # 2. Train/Val Split
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train_df.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        X_val_t = torch.tensor(X_val_df.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
        
        input_dim = X.shape[1]
        
        # 3. Model Training
        model = TabularCNN(input_dim=input_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss))
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        
        # Save Model
        ts = int(time.time())
        model_dir = "models/cnn"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"cnn_tabular_{ts}.pt")
        
        # Save session data for prediction (columns info etc)
        metadata_path = os.path.join(model_dir, f"cnn_metadata_{ts}.joblib")
        
        # Save background data for SHAP (subset of transformed X_train)
        background_path = os.path.join(model_dir, f"cnn_background_{ts}.joblib")
        background_data = X_train_t[:50] # First 50 samples
        joblib.dump(background_data, background_path)
        
        metadata = {
            "input_dim": input_dim,
            "columns": list(X.columns),
            "label_col": label_col,
            "background_path": background_path,
            "encoders": encoders,
            "scaler": scaler,
            "cat_cols": cat_cols,
            "num_cols": num_cols
        }
        
        torch.save(model.state_dict(), model_path)
        joblib.dump(metadata, metadata_path)
        
        # 5. Model Interpretability (Optional: Log summary)
        logger.info("Interpretability metadata saved for SHAP integration")
        
        return {
            "success": True,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "history": history,
            "session_id": session_id,
            "background_path": background_path
        }
        
    finally:
        cleanup_etl_session(session_id)

def predict_cnn_tabular(model_path: str, metadata_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    CNN Prediction for Tabular Data (Regression)
    """
    metadata = joblib.load(metadata_path)
    model = TabularCNN(input_dim=metadata["input_dim"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Create DataFrame from input data
    df = pd.DataFrame([data])
    
    # Apply stateful preprocessing
    encoders = metadata.get("encoders", {})
    scaler = metadata.get("scaler")
    cat_cols = metadata.get("cat_cols", [])
    num_cols = metadata.get("num_cols", [])
    
    # 1. Encode categorical
    for col in cat_cols:
        if col in df.columns:
            # Handle unseen labels by mapping them to the first class or a default if necessary
            # For simplicity, we assume strings are present in training
            le = encoders[col]
            # Convert to string and handle potential unseen categories
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
    # 2. Scale numeric
    if scaler and num_cols:
        df[num_cols] = scaler.transform(df[num_cols])
            
    # Ensure all required columns are present and in order
    for col in metadata["columns"]:
        if col not in df.columns:
            df[col] = 0
            
    X = df[metadata["columns"]]
    
    try:
        X_t = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
    except Exception as e:
        logger.error(f"Tensor conversion failed: {e}")
        # Identify which column is causing the issue
        for col in X.columns:
            try:
                X[col].values.astype(np.float32)
            except Exception:
                logger.error(f"Column '{col}' contains non-numeric data that wasn't encoded. Dtype: {X[col].dtype}, Sample: {X[col].head(1).tolist()}")
        raise ValueError(f"CNN Prediction Error: One or more columns could not be converted to float. Check logs for details.")
    
    with torch.no_grad():
        prediction = model(X_t)
        
    return {
        "prediction": float(prediction.item())
    }
