from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np
import pickle
from datetime import datetime
from config.logger import logger

def prepare_data(csv_path, output_path="processed_data.csv"):
    df = pd.read_csv(csv_path)

    # Target variable
    y = df["price"]

    # Features
    X = df.drop("price", axis=1)

    # Convert yes/no to 1/0
    yes_no_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                   "airconditioning", "prefarea"]
    for col in yes_no_cols:
        X[col] = X[col].map({"yes": 1, "no": 0})

    # One-hot encode categorical
    X = pd.get_dummies(X, columns=["furnishingstatus"], drop_first=True)

    # Combine processed X + y to save
    processed_df = X.copy()
    processed_df["price"] = y

    # Save processed CSV
    processed_df.to_csv(output_path, index=False)

    return {
        "status": "success",
        "message": "Data processed and saved successfully",
        "processed_file": output_path,
        "rows": len(processed_df),
        "columns": list(processed_df.columns),
        "sample": processed_df.head(3).to_dict(orient="records")  # small preview in JSON
    }
def train_model(processed_csv_path, model_dir="models"):
    # Load processed data
    df = pd.read_csv(processed_csv_path)

    # Separate X and y
    y = df["price"]
    X = df.drop("price", axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{model_dir}/house_price_model_{timestamp}.pkl"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Return JSON response
    return {
        "success": True,
        "message": "Model trained and saved successfully",
        "model_path": model_path,
        "metrics": {
            "rmse": rmse,
            "r2_score": r2
        },
        "features_used": list[Any](X.columns)
    }
    
def evaluate_model(model_path, X_test, y_test):
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }