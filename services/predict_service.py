# services/predict_service.py
import os
import torch
from torchvision import transforms
from PIL import Image
from services.train_service import build_model, get_device, default_transforms
from config.logger import logger


def load_model(model_path: str, num_classes: int, pretrained: bool = False):
    logger.info(f"Loading model from path: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(model_path)

    device = get_device()

    logger.info(f"Building model with num_classes={num_classes} pretrained={pretrained}")
    model = build_model(num_classes=num_classes, pretrained=pretrained)

    logger.info("Loading model state_dict...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    logger.info("Model moved to device and set to eval mode.")
    model.eval()

    return model


def predict_image(model, image_path: str, img_size: int = 224):
    logger.info(f"Predicting image: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(image_path)

    logger.info(f"Loading image and applying transforms (img_size={img_size})")
    tf = default_transforms(train=False, img_size=img_size)

    img = Image.open(image_path).convert("RGB")
    inp = tf(img).unsqueeze(0)  # batch dim

    device = get_device()
    inp = inp.to(device)

    logger.info("Running inference...")
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_class = int(pred.cpu().item())
    confidence = float(conf.cpu().item())

    logger.info(f"Prediction complete — Class: {pred_class}, Confidence: {confidence:.4f}")

    return {
        "pred_class": pred_class,
        "confidence": confidence
    }
