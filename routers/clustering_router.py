# routers/clustering_router.py

from fastapi import APIRouter, UploadFile, HTTPException
from services.clustering_service import ClusteringService
from config.logger import logger
import pandas as pd
import tempfile
import os

clustering_router = APIRouter(prefix="/clustering", tags=["Clustering"])
service = ClusteringService()


@clustering_router.post("/train")
async def train_model(csv_file: UploadFile, clusters: int = 3):

    logger.info(f"[TRAIN] Training request received. File: {csv_file.filename}, Clusters: {clusters}")

    try:
        # Save uploaded CSV to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        file_bytes = await csv_file.read()
        tmp.write(file_bytes)
        tmp.close()

        logger.info(f"[TRAIN] CSV saved to temporary file: {tmp.name}")

        # Train the clustering model
        result = service.train(tmp.name, n_clusters=clusters)

        logger.info("[TRAIN] Training completed successfully.")

        return result

    except Exception as e:
        logger.error(f"[TRAIN] Error during training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Training failed.")

    finally:
        # Clean up the temp file
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
            logger.debug(f"[TRAIN] Temporary file removed: {tmp.name}")


@clustering_router.post("/predict")
async def predict_cluster(payload: dict):

    logger.info(f"[PREDICT] Prediction request received.")

    if "data" not in payload:
        logger.warning("[PREDICT] Missing 'data' field in request payload.")
        raise HTTPException(status_code=400, detail="Payload must include 'data'.")

    try:
        logger.debug(f"[PREDICT] Payload data: {payload['data']}")

        result = service.predict(
            "models/clustering/autoencoder.pth",
            "models/clustering/kmeans.pkl",
            "models/clustering/transformer.pkl",
            payload["data"]
        )

        logger.info(f"[PREDICT] Prediction successful. Cluster: {result['cluster']}")
        return result

    except FileNotFoundError as fnf:
        logger.error(f"[PREDICT] Model files missing: {fnf}", exc_info=True)
        raise HTTPException(status_code=500, detail="Model files missing. Train a model first.")

    except Exception as e:
        logger.error(f"[PREDICT] Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed.")
