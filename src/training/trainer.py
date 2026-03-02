"""
CNN Training with:
✔ Train / Validation split
✔ Proper pos_weight from train only
✔ Best model saving
✔ Validation metrics tracking
"""

import torch

torch.set_num_threads(4)

import mlflow
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split

from src.data.data_loader import ChestXrayDataset
from src.models.cnn_model import ChestXrayCNN


def train_cnn():

    # ─────────────────────────────────────────────
    # 1️⃣ Load Dataset
    # ─────────────────────────────────────────────

    full_dataset = ChestXrayDataset(
        csv_file="data/processed/subset.csv", image_dir="data/processed/images"
    )

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0  # safer for 8GB RAM
    )

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = ChestXrayCNN(num_classes=14)

    # ─────────────────────────────────────────────
    # 2️⃣ Compute pos_weight USING TRAIN SET ONLY
    # ─────────────────────────────────────────────

    print("Computing class imbalance weights (train only)...")

    all_labels = []

    for _, labels in train_loader:
        all_labels.append(labels.numpy())

    all_labels = np.vstack(all_labels)

    class_counts = all_labels.sum(axis=0)
    num_samples = len(all_labels)

    pos_weight = (num_samples - class_counts) / (class_counts + 1e-6)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    print("Train pos_weight:", pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ─────────────────────────────────────────────
    # 3️⃣ Training + Validation Loop
    # ─────────────────────────────────────────────

    best_f1 = 0

    mlflow.set_experiment("ChestXray-CNN-TrainVal")

    with mlflow.start_run():

        for epoch in range(10):

            # ---------------- TRAIN ----------------
            model.train()
            total_loss = 0

            for images, labels in train_loader:

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # ---------------- VALIDATION ----------------
            model.eval()

            all_preds = []
            all_true = []

            with torch.no_grad():
                for images, labels in val_loader:

                    outputs = model(images)
                    probs = torch.sigmoid(outputs)

                    threshold = 0.55
                    preds = (probs > threshold).int()

                    all_preds.extend(preds.numpy())
                    all_true.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_true = np.array(all_true)

            f1_micro = f1_score(all_true, all_preds, average="micro", zero_division=0)
            f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)

            precision = precision_score(
                all_true, all_preds, average="micro", zero_division=0
            )
            recall = recall_score(all_true, all_preds, average="micro", zero_division=0)

            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val F1 Micro: {f1_micro:.4f}")
            print(f"Val F1 Macro: {f1_macro:.4f}")
            print(f"Val Precision: {precision:.4f}")
            print(f"Val Recall: {recall:.4f}")

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_f1_micro", f1_micro, step=epoch)

            # Save best model
            if f1_micro > best_f1:
                best_f1 = f1_micro
                torch.save(model.state_dict(), "best_model.pth")
                print("Best model saved.")

    print("\nTraining completed.")


if __name__ == "__main__":
    train_cnn()
