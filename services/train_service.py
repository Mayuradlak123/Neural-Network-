# services/train_service.py
import os
import time
from datetime import datetime
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np

from services.torch_service import ImageCSVLoader
from sklearn.metrics import accuracy_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def default_transforms(train: bool = True, img_size: int = 224):
    from torchvision import transforms
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

def build_model(num_classes: int, pretrained: bool = True):
    model = models.resnet18(pretrained=pretrained)
    # replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train(
    train_csv: str,
    val_csv: str,
    model_dir: str = "models",
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    img_size: int = 224,
    pretrained: bool = True,
    num_workers: int = 2
) -> Dict[str, Any]:
    """
    Train a ResNet18 classifier on CSV dataset and save .pt model.
    Returns JSON-like dict with metrics and model path.
    """
    os.makedirs(model_dir, exist_ok=True)
    device = get_device()

    # transforms
    train_tf = default_transforms(train=True, img_size=img_size)
    val_tf = default_transforms(train=False, img_size=img_size)

    # datasets & loaders
    train_ds = ImageCSVLoader(train_csv, transform=train_tf)
    val_ds = ImageCSVLoader(val_csv, transform=val_tf)

    # infer num classes from labels present
    labels_all = np.unique(train_ds.df["label"].astype(int).values)
    num_classes = int(labels_all.max()) + 1  # assumes labels are 0..C-1; adjust if needed

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_ds)
        history["train_loss"].append(epoch_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        preds = []
        targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, pred_cls = torch.max(outputs, 1)
                preds.extend(pred_cls.cpu().numpy().tolist())
                targets.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = val_loss / len(val_ds)
        history["val_loss"].append(epoch_val_loss)

        epoch_val_acc = accuracy_score(targets, preds)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # Save best
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"resnet18_best_{timestamp}.pt"
            best_model_path = os.path.join(model_dir, model_filename)
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch}/{epochs} — train_loss: {epoch_train_loss:.4f} val_loss: {epoch_val_loss:.4f} val_acc: {epoch_val_acc:.4f}")

    # final save (if no best saved)
    if best_model_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(model_dir, f"resnet18_final_{timestamp}.pt")
        torch.save(model.state_dict(), best_model_path)

    return {
        "success": True,
        "model_path": best_model_path,
        "history": history,
        "best_val_acc": best_val_acc,
        "device": str(device)
    }
