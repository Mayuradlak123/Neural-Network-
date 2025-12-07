# services/clustering_service.py

import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple
from config.logger import logger


# =======================================================
# AUTOENCODER
# =======================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        logger.debug(f"Initializing AutoEncoder with input_dim={input_dim}, latent_dim={latent_dim}")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


# =======================================================
# ETL HELPERS
# =======================================================
def _detect_identifier_columns(df: pd.DataFrame) -> List[str]:
    """Detect ID columns so clustering is not biased."""
    id_cols = []
    for c in df.columns:
        cname = c.lower().strip()
        if "id" in cname:
            id_cols.append(c)
    return id_cols


def etl_clean(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
    drop_identifier_columns: bool = True
) -> pd.DataFrame:

    logger.info("[ETL] Starting cleaning")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Drop duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    logger.info(f"[ETL] Removed {before - df.shape[0]} duplicates")

    # Drop user-specified columns
    if drop_columns:
        drop_cols = [c for c in drop_columns if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info(f"[ETL] Dropped user columns: {drop_cols}")

    # Drop ID columns
    if drop_identifier_columns:
        id_cols = _detect_identifier_columns(df)
        if id_cols:
            df = df.drop(columns=id_cols)
            logger.info(f"[ETL] Dropped identifier columns: {id_cols}")

    # Fill Missing
    for col in df.columns:
        if df[col].dtype == "object":
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
            df[col] = df[col].fillna(mode_val)
        else:
            df[col] = df[col].fillna(df[col].median())

    logger.info("[ETL] Completed")
    return df


# =======================================================
# FEATURE ENGINEERING
# =======================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[FE] Starting feature engineering")
    df = df.copy()

    # Boolean Cleanup
    bool_map = {"True": 1, "False": 0, "true": 1, "false": 0, "Yes": 1, "No": 0}
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace(bool_map)

    # log-scaling salary or similar
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if "salary" in col.lower():
            df[f"log_{col}"] = np.log1p(df[col]).clip(lower=0)

    # Age binning
    if "Age" in df.columns:
        df["Age"] = df["Age"].astype(float)
        df["age_bin"] = pd.cut(df["Age"], bins=[-1, 25, 40, 60, 150], labels=[0, 1, 2, 3]).astype(int)

    logger.info("[FE] Completed")
    return df


# =======================================================
# CLUSTERING SERVICE
# =======================================================
class ClusteringService:

    def __init__(self):
        self.model_dir = "models/clustering"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"ClusteringService initialized. Model directory: {self.model_dir}")

    # ----------------------------------------------------
    # TRAIN
    # ----------------------------------------------------
    def train(self, csv_path: str, n_clusters: int = 3):
        logger.info(f"[CLUSTER TRAIN] CSV={csv_path}, clusters={n_clusters}")

        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded: {df.shape}")

        # --------- ETL ---------
        df = etl_clean(df)

        # --------- Feature Engineering ---------
        df = feature_engineering(df)

        # Keep numeric only for clustering
        X = df.select_dtypes(include=["int64", "float64"]).copy()
        logger.info(f"Features selected for clustering: {list(X.columns)}")

        # -------- Scaling --------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---------- AutoEncoder ----------
        input_dim = X_scaled.shape[1]
        model = AutoEncoder(input_dim=input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        logger.info("Training AutoEncoder for deep clustering...")
        for epoch in range(50):
            optimizer.zero_grad()
            output, _ = model(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss={loss.item():.6f}")

        # Extract Latent Space
        with torch.no_grad():
            _, latent = model(X_tensor)

        latent_np = latent.numpy()

        # --------- KMeans ---------
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(latent_np)

        # --------- Save Models ---------
        ae_path = f"{self.model_dir}/autoencoder.pth"
        km_path = f"{self.model_dir}/kmeans.pkl"
        sc_path = f"{self.model_dir}/scaler.pkl"

        torch.save(model.state_dict(), ae_path)
        pickle.dump(kmeans, open(km_path, "wb"))
        pickle.dump(scaler, open(sc_path, "wb"))

        logger.info("Deep clustering model saved successfully")

        return {
            "success": True,
            "message": "Clustering model trained successfully.",
            "paths": {
                "autoencoder": ae_path,
                "kmeans": km_path,
                "scaler": sc_path
            },
            "clusters": n_clusters
        }

    # ----------------------------------------------------
    # PREDICT
    # ----------------------------------------------------
    def predict(self, model_path_ae, model_path_km, model_path_scaler, data: dict):

        logger.info(f"[CLUSTER PREDICT] Input: {data}")

        df = pd.DataFrame([data])

        # ETL + FE same as training
        df = etl_clean(df, drop_identifier_columns=False)
        df = feature_engineering(df)

        # Numeric Only
        X = df.select_dtypes(include=["int64", "float64"]).copy()

        # Load Models
        scaler = pickle.load(open(model_path_scaler, "rb"))
        kmeans = pickle.load(open(model_path_km, "rb"))

        X_scaled = scaler.transform(X)

        input_dim = X_scaled.shape[1]
        model = AutoEncoder(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path_ae))
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            _, latent = model(X_tensor)

        cluster = kmeans.predict(latent.numpy())

        return {"success": True, "cluster": int(cluster[0])}
