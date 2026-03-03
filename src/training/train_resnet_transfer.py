"""
Transfer Learning Training Script
Uses ResNetTransfer model from models/
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, random_split

from src.data.data_loader import ChestXrayDataset
from src.models.resnet_model import ResNetTransfer


def convert_to_binary(labels):
    return (labels.sum(dim=1) > 0).float().unsqueeze(1)


def train_resnet(mode="frozen"):

    print(f"\nTraining ResNet18 - Mode: {mode}")

    full_dataset = ChestXrayDataset(
        csv_file="data/processed/subset.csv", image_dir="data/processed/images"
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = ResNetTransfer(mode=mode)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4 if mode == "finetune" else 5e-4,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    epochs = 15
    best_f1 = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(epochs):

        model.train()
        total_train_loss = 0

        for images, labels in train_loader:
            labels = convert_to_binary(labels)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_true = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = convert_to_binary(labels)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_preds.extend(preds.numpy())
                all_true.extend(labels.numpy())
                all_probs.extend(probs.numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        f1 = f1_score(all_true, all_preds)
        precision = precision_score(all_true, all_preds)
        recall = recall_score(all_true, all_preds)
        roc_auc = roc_auc_score(all_true, all_probs)

        val_f1_scores.append(f1)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val F1: {f1:.4f}")
        print(f"Val Precision: {precision:.4f}")
        print(f"Val Recall: {recall:.4f}")
        print(f"Val ROC-AUC: {roc_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_resnet_{mode}.pth")
            print("✅ Best model saved")
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print("⛔ Early stopping triggered")
            break

    print("\nTraining completed.")


if __name__ == "__main__":
    train_resnet(mode="frozen")
    train_resnet(mode="finetune")
