# routes/ml_routes.py
import os
import zipfile
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from services.torch_service import ImageCSVLoader
from services.train_service import train, get_device
from services.predict_service import load_model, predict_image
from config.logger import logger


# ---------------------------------------------------------
# 🔥 Combined Router (Torch + ML)
# ---------------------------------------------------------
torch_router = APIRouter(tags=["ML & Torch"])


# =========================================================
# 🧩 TORCH PART — Load Images From CSV
# =========================================================

class ImageCSVRequest(BaseModel):
    csv_path: str
    batch_size: Optional[int] = 2
    shuffle: Optional[bool] = True


@torch_router.post("/load-images")
def load_images_api(request: ImageCSVRequest):
    """
    Load images & labels from CSV using PyTorch Dataset + DataLoader
    Returns FIRST batch only.
    """

    logger.info(f"Received request to load CSV: {request.csv_path}")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    try:
        dataset = ImageCSVLoader(request.csv_path, transform=transform)
        loader = DataLoader(dataset, batch_size=request.batch_size, shuffle=request.shuffle)

        for images, labels in loader:
            batch_info = {
                "batch_image_shape": list(images.shape),
                "batch_labels": labels.tolist()
            }
            return {
                "data": batch_info,
                "success": True,
                "statuc_code": 200,
                "message": "Image processed success"
            }

    except Exception as e:
        logger.error(f"Error in /load-images API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# =========================================================
# 🧩 ML PIPELINE PART — Upload CSV, Upload Images, Train, Predict
# =========================================================

# Helper: save uploaded file
def save_upload_file(upload_file: UploadFile, dest_path: str):
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)


# -------------- Upload CSV --------------------------
@torch_router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    dest = os.path.join("data", file.filename)
    save_upload_file(file, dest)
    return {"success": True, "path": dest}


# -------------- Upload ZIP of Images --------------------------
@torch_router.post("/upload-images-zip")
async def upload_images_zip(file: UploadFile = File(...)):
    """
    Upload ZIP → Extract → Auto-generate CSV with image paths + dummy label (0)
    Returns: extracted folder + CSV path
    """ 
    try:
        os.makedirs("data/images", exist_ok=True)
        os.makedirs("data/uploads", exist_ok=True)

        # Save uploaded ZIP temporarily
        dest_zip = os.path.join("data", file.filename)
        save_upload_file(file, dest_zip)

        # Extract folder name
        folder_name = os.path.splitext(file.filename)[0]
        extract_dir = os.path.join("data/images", folder_name)
        os.makedirs(extract_dir, exist_ok=True)

        # Extract ZIP
        with zipfile.ZipFile(dest_zip, 'r') as z:
            z.extractall(extract_dir)

        # Remove ZIP after extraction
        os.remove(dest_zip)

        # -----------------------------------------
        # Create CSV automatically
        # -----------------------------------------
        csv_path = os.path.join("data/uploads", f"{folder_name}.csv")

        rows = []
        for root, dirs, files in os.walk(extract_dir):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(root, fname).replace("\\", "/")
                    rows.append([img_path, 0])  # default label = 0

        # Write CSV
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            writer.writerows(rows)

        return {
            "success": True,
            "extracted_to": extract_dir,
            "csv_path": csv_path,
            "total_images": len(rows),
            "message": "Images extracted & CSV created"
        }

    except Exception as e:
        logger.error(f"Error processing zip: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------- TRAIN MODEL --------------------------
class TrainRequest(BaseModel):
    train_csv_path: str
    val_csv_path: str
    batch_size: Optional[int] = 16
    epochs: Optional[int] = 5
    lr: Optional[float] = 1e-3
    img_size: Optional[int] = 224


@torch_router.post("/train")
def train_api(req: TrainRequest):
    try:
        result = train(
            train_csv=req.train_csv_path,
            val_csv=req.val_csv_path,
            batch_size=req.batch_size,
            epochs=req.epochs,
            lr=req.lr,
            img_size=req.img_size
        )
        return {"success": True, "result": result}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# -------------- PREDICT --------------------------
class PredictRequest(BaseModel):
    model_path: str
    num_classes: int
    image_path: str


@torch_router.post("/predict")
def predict_api(req: PredictRequest):
    try:
        model = load_model(req.model_path, num_classes=req.num_classes)
        out = predict_image(model, req.image_path)
        return {"success": True, "prediction": out}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    