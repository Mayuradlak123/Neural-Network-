# services/predict_service.py
import os
import torch
from torchvision import transforms
from PIL import Image
from services.train_service import build_model, get_device, default_transforms

def load_model(model_path: str, num_classes: int, pretrained: bool = False):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    device = get_device()
    model = build_model(num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path: str, img_size: int = 224):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    tf = default_transforms(train=False, img_size=img_size)
    img = Image.open(image_path).convert("RGB")
    inp = tf(img).unsqueeze(0)  # batch dim
    device = get_device()
    inp = inp.to(device)
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return {"pred_class": int(pred.cpu().item()), "confidence": float(conf.cpu().item())}
