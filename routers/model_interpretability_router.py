from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.model_interpretability_service import InterpretabilityService

model_interpretability_router = APIRouter(prefix="/interpret", tags=["Interpretability"])
service = InterpretabilityService()


# -------------------------------
# Request Models
# -------------------------------
class SHAPRequest(BaseModel):
    model_path: str
    scaler_path: str
    data: dict


class LIMERequest(BaseModel):
    model_path: str
    scaler_path: str
    dataset_csv: str
    data: dict


# -------------------------------
# SHAP API Route
# -------------------------------
@model_interpretability_router.post("/shap")
async def shap_explanation(req: SHAPRequest):
    """
    SHAP → Shows feature contribution for prediction
    """
    try:
        result = service.shap_explain(
            model_path=req.model_path,
            scaler_path=req.scaler_path,
            data=req.data
        )

        return {
            "status": "success",
            "message": "SHAP explanation generated successfully.",
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to generate SHAP explanation.",
                "error": str(e)
            }
        )


# -------------------------------
# LIME API Route
# -------------------------------
@model_interpretability_router.post("/lime")
async def lime_explanation(req: LIMERequest):
    """
    LIME → Gives local explanation around a single prediction
    """
    try:
        result = service.lime_explain(
            model_path=req.model_path,
            scaler_path=req.scaler_path,
            dataset_csv=req.dataset_csv,
            data=req.data
        )

        return {
            "status": "success",
            "message": "LIME explanation generated successfully.",
            "data": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to generate LIME explanation.",
                "error": str(e)
            }
        )
