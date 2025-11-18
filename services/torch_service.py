import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from config.logger import logger


class ImageCSVLoader(Dataset):
    """
    Loads images and labels from a CSV file using PyTorch Dataset.
    CSV must contain: image_path, label
    """
    def __init__(self, csv_path: str, transform=None):
        logger.info(f"Loading CSV file from: {csv_path}")

        if not os.path.exists(csv_path):
            logger.error(f"CSV not found: {csv_path}")
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)       # <-- ADD THIS

        if "image_path" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")

        logger.info(f"CSV loaded - Total rows: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]

        logger.info(f"Loading image at index {idx}: {image_path}")

        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
