from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from services.etl_feature_service import get_etl_session, cleanup_etl_session
from config.logger import logger
import uuid
from services.ETL import prepare_data,train_model,evaluate_model
etl_router = APIRouter(tags=["etl"])
ml_router=APIRouter(tags=["ml","models","learning"])
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
from rules.index import RemoveOutliersRequest,EvaluateRequest, LoadCSVRequest,ExportDataRequest,ExportDataRequest,CreateFeaturesRequest ,ScaleFeaturesRequest,PredictRequest,MissingValuesRequest,EncodeCategoricalRequest
import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
@ml_router.post("/preprocess-data")
async def preprocess_data(request: LoadCSVRequest):
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output file path dynamically
        output_path = f"data/processed_{timestamp}.csv"

        # Call preprocessing function
        response = prepare_data(csv_path=request.file_path, output_path=output_path)

        return response

    except Exception as e:
        return {
            "success": False,
            "status_code": 400,
            "message": f"Failed to process data: {str(e)}"
        }
@ml_router.post("/train-model")
async def train_model_api(request: LoadCSVRequest):
    try:
        result = train_model(request.file_path)
        return result
    except Exception as e:
        return {
            "success": False,
            "message": f"Training failed: {str(e)}"
        }
@ml_router.post("/predict-price")
async def predict_price(req: PredictRequest):
    try:
        # Load trained model
        model = load_model(req.model_path)

        # Convert request to dataframe-like structure
        features = {
            "area": req.area,
            "bedrooms": req.bedrooms,
            "bathrooms": req.bathrooms,
            "stories": req.stories,
            "mainroad": req.mainroad,
            "guestroom": req.guestroom,
            "basement": req.basement,
            "hotwaterheating": req.hotwaterheating,
            "airconditioning": req.airconditioning,
            "parking": req.parking,
            "prefarea": req.prefarea,
            "furnishingstatus_semi-furnished": req.furnishingstatus_semi_furnished,
            "furnishingstatus_unfurnished": req.furnishingstatus_unfurnished
        }

        # Convert to a 2D array
        input_df = pd.DataFrame([features])

        # Predict
        predicted_price = model.predict(input_df)[0]

        return {
            "success": True,
            "predicted_price": float(predicted_price),
            "features_used": features
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Prediction failed: {str(e)}"
        }


@ml_router.post("/evaluate-model")
async def evaluate_model_api(request: EvaluateRequest):
    try:
        # Load test CSV
        df = pd.read_csv(request.test_data_path)

        # Separate independent & dependent variables
        y_test = df["price"]
        X_test = df.drop("price", axis=1)

        # Convert yes/no
        yes_no_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
        for col in yes_no_cols:
            if col in X_test.columns:
                X_test[col] = X_test[col].map({"yes": 1, "no": 0})

        # One-hot encode furnishingstatus
        if "furnishingstatus" in X_test.columns:
            X_test = pd.get_dummies(X_test, columns=["furnishingstatus"], drop_first=True)

        # Evaluate model
        metrics = evaluate_model(
            model_path=request.model_path,
            X_test=X_test,
            y_test=y_test
        )

        return {
            "success": True,
            "status_code": 200,
            "metrics": metrics
        }

    except Exception as e:
        return {
            "success": False,
            "status_code": 400,
            "message": f"Evaluation failed: {str(e)}"
        }

@etl_router.post("/load-csv")
async def load_csv_file(request: LoadCSVRequest):
    """
    Load CSV file and start ETL session
    """
    try:
        logger.info(f"Loading CSV from: {request.file_path}")
        
        # Create new session
        session_id = str(uuid.uuid4())
        etl_service = get_etl_session(session_id)
        
        # Load CSV
        result = etl_service.load_csv(request.file_path)
        
        return {
            **result,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/handle-missing")
async def handle_missing_values(request: MissingValuesRequest):
    """
    Handle missing values in the dataset
    """
    try:
        etl_service = get_etl_session(request.session_id)
        result = etl_service.handle_missing_values(
            strategy=request.strategy,
            columns=request.columns
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to handle missing values: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/encode-categorical")
async def encode_categorical(request: EncodeCategoricalRequest):
    """
    Encode categorical variables
    """
    try:
        etl_service = get_etl_session(request.session_id)
        result = etl_service.encode_categorical(
            columns=request.columns,
            method=request.method
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to encode categorical: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/scale-features")
async def scale_features(request: ScaleFeaturesRequest):
    """
    Scale numeric features
    """
    try:
        etl_service = get_etl_session(request.session_id)
        result = etl_service.scale_features(
            columns=request.columns,
            method=request.method
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to scale features: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/create-features")
async def create_features(request: CreateFeaturesRequest):
    """
    Create new features using feature engineering
    """
    try:
        etl_service = get_etl_session(request.session_id)
        result = etl_service.create_features(operations=request.operations)
        return result
        
    except Exception as e:
        logger.error(f"Failed to create features: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/remove-outliers")
async def remove_outliers(request: RemoveOutliersRequest):
    """
    Remove outliers from specified columns
    """
    try:
        etl_service = get_etl_session(request.session_id)
        result = etl_service.remove_outliers(
            columns=request.columns,
            method=request.method,
            threshold=request.threshold
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to remove outliers: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/export")
async def export_data(request: ExportDataRequest):
    """
    Export processed data to CSV
    """
    try:
        etl_service = get_etl_session(request.session_id)
        result = etl_service.export_data(output_path=request.output_path)
        return result
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.get("/state/{session_id}")
async def get_current_state(session_id: str):
    """
    Get current state of the dataframe
    """
    try:
        etl_service = get_etl_session(session_id)
        result = etl_service.get_current_state()
        return result
        
    except Exception as e:
        logger.error(f"Failed to get state: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.get("/history/{session_id}")
async def get_transformations_history(session_id: str):
    """
    Get history of all transformations
    """
    try:
        etl_service = get_etl_session(session_id)
        history = etl_service.get_transformations_history()
        return {
            "success": True,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.post("/reset/{session_id}")
async def reset_to_original(session_id: str):
    """
    Reset data to original state
    """
    try:
        etl_service = get_etl_session(session_id)
        result = etl_service.reset_to_original()
        return result
        
    except Exception as e:
        logger.error(f"Failed to reset: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@etl_router.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """
    Cleanup ETL session
    """
    try:
        cleanup_etl_session(session_id)
        return {
            "success": True,
            "message": "Session cleaned up successfully"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup session: {e}")
        raise HTTPException(status_code=500, detail=str(e))