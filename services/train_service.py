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
from config.logger import logger


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def default_transforms(train: bool = True, img_size: int = 224):
    from torchvision import transforms

    logger.info(f"Creating {'training' if train else 'validation'} transforms with image size {img_size}")

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
    logger.info(f"Building ResNet18 model — pretrained={pretrained}, num_classes={num_classes}")
    model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    logger.info("Model built successfully.")
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

    logger.info("Starting training process...")
    os.makedirs(model_dir, exist_ok=True)
    device = get_device()

    # transforms
    train_tf = default_transforms(train=True, img_size=img_size)
    val_tf = default_transforms(train=False, img_size=img_size)

    # datasets & loaders
    logger.info(f"Loading training dataset: {train_csv}")
    train_ds = ImageCSVLoader(train_csv, transform=train_tf)

    logger.info(f"Loading validation dataset: {val_csv}")
    val_ds = ImageCSVLoader(val_csv, transform=val_tf)

    # number of classes
    labels_all = np.unique(train_ds.data["label"].astype(int).values)
    num_classes = int(labels_all.max()) + 1

    logger.info(f"Detected {num_classes} classes from training dataset.")

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

    logger.info(f"Training for {epochs} epochs.")

    for epoch in range(1, epochs + 1):
        logger.info(f"----- Epoch {epoch}/{epochs} START -----")

        model.train()
        running_loss = 0.0

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

        logger.info(f"Epoch {epoch} - Training Loss: {epoch_train_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        preds, targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, pred_cls = torch.max(outputs, 1)
                preds.extend(pred_cls.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_ds)
        history["val_loss"].append(epoch_val_loss)

        epoch_val_acc = accuracy_score(targets, preds)
        history["val_acc"].append(epoch_val_acc)

        logger.info(
            f"Epoch {epoch} - Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_acc:.4f}"
        )

        scheduler.step(epoch_val_loss)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"resnet18_best_{timestamp}.pt"
            best_model_path = os.path.join(model_dir, model_filename)
            torch.save(model.state_dict(), best_model_path)

            logger.info(f"New best model saved: {best_model_path}")

        logger.info(f"----- Epoch {epoch}/{epochs} END -----")

    if best_model_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(model_dir, f"resnet18_final_{timestamp}.pt")
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"No best model found, saving final model: {best_model_path}")

    logger.info("Training completed successfully!")

    return {
        "success": True,
        "model_path": best_model_path,
        "history": history,
        "best_val_acc": best_val_acc,
        "device": str(device)
    }
