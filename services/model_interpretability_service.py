import pandas as pd
import pickle
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import os
from config.logger import logger

import torch
from torch import nn

class TabularModel(nn.Module):
    def __init__(self, input_dim):
        super(TabularModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)      # regression output
        )

    def forward(self, x):
        return self.layers(x)

class InterpretabilityService:

    def __init__(self):
        self.model_dir = "models/trained"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("InterpretabilityService initialized.")

    # ----------------------------------------
    # Run SHAP Explainability
    # ----------------------------------------
    def shap_explain(self, model_path, scaler_path, data: dict):
        """
        SHAP — feature contributions for neural networks (PyTorch)
        """
        import torch
        import pandas as pd
        import pickle
        import shap

        logger.info("SHAP explanation (Neural Network) started")

        df = pd.DataFrame([data])

        # Load scaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded: {scaler_path}")

        # Scale input
        X = df.select_dtypes(include=["int64", "float64"])
        X_scaled = scaler.transform(X)
        input_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Load model
        input_dim = X_scaled.shape[1]
        model = TabularModel(input_dim=input_dim)
        state = torch.load(model_path, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        logger.info(f"Model loaded: {model_path}")

        # Background samples for SHAP
        background = torch.randn(50, input_dim)

        # Deep SHAP
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(input_tensor)

        feature_list = list(X.columns)
        feature_contributions = dict(zip(feature_list, shap_values[0][0].tolist()))

        pred = float(model(input_tensor).detach().numpy()[0][0])

        logger.info("SHAP explanation completed")

        return {
            "success": True,
            "method": "SHAP",
            "prediction": pred,
            "feature_contributions": feature_contributions
        }

    # ----------------------------------------
    # Run LIME Explainability
    # ----------------------------------------
    def lime_explain(self, model_path, scaler_path, dataset_csv, data: dict):
        """
        LIME ka matlab — ek prediction ke aas-paas local explanation
        """

        logger.info("LIME explanation started")

        # DataFrame
        df = pd.DataFrame([data])

        # Dataset for Lime Training reference
        full_df = pd.read_csv(dataset_csv)

        # Select only numeric
        X_all = full_df.select_dtypes(include=["int64", "float64"])
        X = df.select_dtypes(include=["int64", "float64"])

        scaler = pickle.load(open(scaler_path, "rb"))
        model = pickle.load(open(model_path, "rb"))

        X_scaled = scaler.transform(X)
        X_all_scaled = scaler.fit_transform(X_all)

        # Lime Explainer initialization
        explainer = LimeTabularExplainer(
            X_all_scaled,
            feature_names=list(X.columns),
            verbose=True,
            mode="regression"  # Classification me "classification" karna hota
        )

        explanation = explainer.explain_instance(
            X_scaled[0], 
            model.predict,
            num_features=5
        )

        logger.info("LIME explanation completed")

        return {
            "success": True,
            "method": "LIME",
            "prediction": float(model.predict(X_scaled)[0]),
            "contributions": dict(explanation.as_list())
        }
