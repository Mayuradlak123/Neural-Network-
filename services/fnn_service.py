
import os
import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.logger import logger

class SimpleFNN(nn.Module):
    """
    A very simple Feedforward Neural Network (FNN) for regression.
    Structure: Input -> Hidden (16) -> ReLU -> Output (1)
    """
    def __init__(self, input_dim: int):
        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_simple_fnn(csv_path: str, target_col: str, epochs: int = 20, lr: float = 0.01):
    """
    Loads a CSV, trains a simple FNN model, and saves it.
    """
    logger.info(f"Starting Simple FNN training on {csv_path}")
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    
    # 2. Preprocessing (Simple: Drop NaNs, Encode Cats, Scale Numerics)
    df = df.dropna()
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical columns (simple label encoding)
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # 4. Initialize Model
    input_dim = X.shape[1]
    model = SimpleFNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Training Loop
    model.train()
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # 6. Save Model & Scaler
    save_dir = "models/fnn"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    
    model_path = os.path.join(save_dir, f"fnn_model_{timestamp}.pt")
    scaler_path = os.path.join(save_dir, f"scaler_{timestamp}.pkl")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim
    }, model_path)
    
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return {
        "success": True,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "final_loss": history[-1],
        "history": history
    }

def predict_simple_fnn(model_path: str, scaler_path: str, input_data: list):
    """
    Loads a saved FNN model and predicts on new data.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler file not found.")
        
    # Load Scaler
    scaler = joblib.load(scaler_path)
    
    # Load Model
    checkpoint = torch.load(model_path)
    input_dim = checkpoint['input_dim']
    model = SimpleFNN(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Preprocess Input
    input_array = np.array(input_data)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
        
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return prediction.numpy().tolist()
