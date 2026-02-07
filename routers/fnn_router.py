
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.fnn_service import train_simple_fnn, predict_simple_fnn

fnn_router = APIRouter(tags=["Simple FNN"])

class TrainFNNRequest(BaseModel):
    csv_path: str
    target_col: str
    epochs: int = 20
    lr: float = 0.01

@fnn_router.post("/train")
async def train_fnn_endpoint(request: TrainFNNRequest):
    try:
        result = train_simple_fnn(
            csv_path=request.csv_path,
            target_col=request.target_col,
            epochs=request.epochs,
            lr=request.lr
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictFNNRequest(BaseModel):
    model_path: str
    scaler_path: str
    input_data: list

@fnn_router.post("/predict")
async def predict_fnn_endpoint(request: PredictFNNRequest):
    try:
        prediction = predict_simple_fnn(
            model_path=request.model_path,
            scaler_path=request.scaler_path,
            input_data=request.input_data
        )
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
