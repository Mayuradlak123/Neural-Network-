from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.train_tabular_service import train_tabular,predict_tabular
# Artificial neural network
ann_router = APIRouter(tags=["Artificial Neural Network"])


class TrainTabularRequest(BaseModel):
    csv_path: str
    label_col: str
    epochs: int = 50
    lr: float = 0.001
    save_processed: bool = False

@ann_router.post("/train-tabular")
def train_tabular_api(req: TrainTabularRequest):
    try:
        result = train_tabular(
            csv_path=req.csv_path,
            label_col=req.label_col,
            epochs=req.epochs,
            lr=req.lr,
            save_processed=req.save_processed,
        )

        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictTabularRequest(BaseModel):
    model_path: str
    data: dict


@ann_router.post("/predict-tabular")
def predict_tabular_api(req: PredictTabularRequest):
    try:
        res = predict_tabular(
            model_path=req.model_path,
            data=req.data
        )
        return {"success": True, "prediction": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
