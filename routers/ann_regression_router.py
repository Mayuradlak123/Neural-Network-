from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config.logger import logger
from services.train_regression_service import train_regression, predict_regression

reg_router = APIRouter(tags=["ANN Regression"])


# ------------------------------
# REQUEST MODELS
# ------------------------------
class TrainRegressionRequest(BaseModel):
    csv_path: str
    label_col: str
    epochs: int = 50
    lr: float = 0.001


class PredictRegressionRequest(BaseModel):
    model_path: str
    data: dict


# ------------------------------
# TRAIN API
# ------------------------------
@reg_router.post("/train-regression")
def train_regression_api(req: TrainRegressionRequest):
    logger.info("===== /train-regression endpoint hit =====")
    logger.debug(f"Request payload: {req.dict()}")

    try:
        logger.info("Starting regression training...")
        res = train_regression(
            csv_path=req.csv_path,
            label_col=req.label_col,
            epochs=req.epochs,
            lr=req.lr
        )

        logger.info("Training completed successfully.")

        return {
            "success": True,
            "message": "Regression model training completed successfully.",
            "stats": {
                "epochs": req.epochs,
                "learning_rate": req.lr,
                "final_train_loss": res["history"]["train_loss"][-1],
                "final_val_loss": res["history"]["val_loss"][-1],
            },
            "result": res
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
# PREDICT API
# ------------------------------
@reg_router.post("/predict-regression")
def predict_regression_api(req: PredictRegressionRequest):
    logger.info("===== /predict-regression endpoint hit =====")
    logger.debug(f"Request payload: {req.dict()}")

    try:
        logger.info("Running regression prediction...")
        res = predict_regression(
            model_path=req.model_path,
            data=req.data
        )

        logger.info("Prediction successful.")

        return {
            "success": True,
            "message": "Prediction generated successfully.",
            "stats": {
                "model_path": req.model_path,
                "input_features_count": len(req.data),
            },
            "prediction": res
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
