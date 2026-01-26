from fastapi import APIRouter, HTTPException
from rules.index import CNNTrainRequest, CNNPredictRequest
from services.cnn_service import train_cnn_tabular, predict_cnn_tabular
from config.logger import logger

cnn_router = APIRouter(tags=["CNN"])

@cnn_router.post("/train")
async def train_cnn(request: CNNTrainRequest):
    """
    Endpoint to train a CNN on tabular data with ETL and FE rule chain
    """
    try:
        result = train_cnn_tabular(
            file_path=request.file_path,
            label_col=request.label_col,
            epochs=request.epochs,
            lr=request.lr,
            batch_size=request.batch_size,
            save_processed=request.save_processed
        )
        return {
            "success": True,
            "message": "CNN training completed successfully",
            "data": result
        }
    except Exception as e:
        logger.error(f"CNN Training Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@cnn_router.post("/predict")
async def predict_cnn(request: CNNPredictRequest):
    """
    Endpoint to run prediction using a trained CNN model (Regression)
    """
    try:
        result = predict_cnn_tabular(
            model_path=request.model_path,
            metadata_path=request.metadata_path,
            data=request.data
        )
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"CNN Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
