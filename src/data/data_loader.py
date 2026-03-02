"""
Data Loader for NIH Chest X-ray Dataset (ChestXray14)

Handles:
- Multi-label classification (14 diseases)
- Image preprocessing
- Safe loading
- Multi-hot label encoding
"""

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ChestXrayDataset(Dataset):
    """
    Custom Dataset for NIH Chest X-ray images
    Converts multi-label disease strings into multi-hot vectors
    """

    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file (subset.csv recommended)
            image_dir (str): Directory containing images
            transform (callable, optional): Image transformations
        """

        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

        # List of 14 disease labels in NIH dataset
        self.all_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia"
        ]

        # Create label to index mapping
        self.label_map = {label: idx for idx, label in enumerate(self.all_labels)}

        # Default transforms
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def encode_labels(self, label_string):
        """
        Convert 'Pneumonia|Edema' → multi-hot tensor of size 14
        """

        multi_hot = torch.zeros(len(self.all_labels))

        if label_string != "No Finding":
            labels = label_string.split("|")

            for label in labels:
                if label in self.label_map:
                    multi_hot[self.label_map[label]] = 1

        return multi_hot

    def __getitem__(self, idx):

        img_name = self.data.iloc[idx]["Image Index"]
        label_string = self.data.iloc[idx]["Finding Labels"]

        img_path = os.path.join(self.image_dir, img_name)

        # Load image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a blank tensor if corrupted
            image = Image.new("RGB", (224, 224))

        image = self.transform(image)

        labels = self.encode_labels(label_string)

        return image, labels