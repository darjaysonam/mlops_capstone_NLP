"""
ANN Training Script (Phase 2 - 2.5)

- Binary classification: Disease vs No Finding
- CNN used as feature extractor
- Custom ANN architecture
- Activation experiments supported
- BatchNorm / Dropout configurable
- L2 Regularization
- Learning Rate Scheduler
- Early stopping
- Plots training curves
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.data_loader import ChestXrayDataset
from src.models.ann_model import ANNClassifier
from src.models.cnn_model import ChestXrayCNN

# -------------------------------------------------
# Feature Extraction using trained CNN
# -------------------------------------------------


def extract_features(cnn_model, dataloader):
    cnn_model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for images, target in tqdm(dataloader):

            outputs = cnn_model(images)

            features.append(outputs)
            labels.append(target)

    features = torch.cat(features)
    labels = torch.cat(labels)

    # Convert multi-label → binary
    labels = (labels.sum(dim=1) > 0).float().unsqueeze(1)

    return features, labels


# -------------------------------------------------
# Training Function
# -------------------------------------------------


def train_ann():

    print("Loading dataset...")

    full_dataset = ChestXrayDataset(
        csv_file="data/processed/subset.csv", image_dir="data/processed/images"
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # -------------------------------------------------
    # Load trained CNN (Feature Extractor)
    # -------------------------------------------------

    print("Loading trained CNN...")
    cnn_model = ChestXrayCNN(num_classes=14)
    cnn_model.load_state_dict(torch.load("best_model.pth"))
    cnn_model.eval()

    print("Extracting training features...")
    train_features, train_labels = extract_features(cnn_model, train_loader)

    print("Extracting validation features...")
    val_features, val_labels = extract_features(cnn_model, val_loader)

    input_dim = train_features.shape[1]
    print("Feature dimension:", input_dim)

    # -------------------------------------------------
    # Choose Experiment Configuration HERE
    # -------------------------------------------------

    model = ANNClassifier(
        input_dim=input_dim,
        hidden1=256,
        hidden2=128,
        activation="tanh",  # change here for experiments
        use_batchnorm=True,  # True / False
        dropout=0.5,  # 0.0 to disable
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4  # L2 regularization
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    epochs = 20
    best_f1 = 0
    patience_counter = 0
    early_stop_patience = 5

    train_losses = []
    val_losses = []
    val_f1_scores = []

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(epochs):

        model.train()

        outputs = model(train_features)
        loss = criterion(outputs, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_labels)

        val_losses.append(val_loss.item())

        probs = torch.sigmoid(val_outputs)
        preds = (probs > 0.5).int()

        f1 = f1_score(val_labels.numpy(), preds.numpy())
        precision = precision_score(val_labels.numpy(), preds.numpy())
        recall = recall_score(val_labels.numpy(), preds.numpy())
        roc_auc = roc_auc_score(val_labels.numpy(), probs.numpy())

        val_f1_scores.append(f1)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {loss.item():.4f}")
        print(f"Val Loss: {val_loss.item():.4f}")
        print(f"Val F1: {f1:.4f}")
        print(f"Val Precision: {precision:.4f}")
        print(f"Val Recall: {recall:.4f}")
        print(f"Val ROC-AUC: {roc_auc:.4f}")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_ann_model.pth")
            print("✅ Best ANN model saved")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("⛔ Early stopping triggered")
            break

    # -------------------------------------------------
    # Plot Curves
    # -------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("ann_loss_curve_tanh.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_f1_scores, label="Validation F1")
    plt.legend()
    plt.title("Validation F1 Curve")
    plt.savefig("ann_f1_curve_tanh.png")
    plt.show()

    print("\nTraining completed.")


if __name__ == "__main__":
    train_ann()
