import argparse
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

# =============================
# Argument Parsing (MLflow Projects compatible)
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size


# =============================
# Dummy Dataset (replace with real data if needed)
# =============================
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000, 5)).float()

train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)


# =============================
# Model Definition
# =============================
class MultiLabelModel(nn.Module):
    def __init__(self, input_dim=20, num_labels=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLabelModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# =============================
# Log Hyperparameters
# =============================
mlflow.log_param("epochs", EPOCHS)
mlflow.log_param("learning_rate", LR)
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("optimizer", "Adam")
mlflow.log_param("loss_function", "BCEWithLogitsLoss")


# =============================
# Training Loop
# =============================
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)

            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    val_loss /= len(val_loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_f1_micro", f1_micro, step=epoch)
    mlflow.log_metric("val_f1_macro", f1_macro, step=epoch)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print("-" * 40)


# =============================
# Confusion Matrix Logging
# =============================
cm = confusion_matrix(all_labels.argmax(axis=1), all_preds.argmax(axis=1))

plt.figure(figsize=(6, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.close()

mlflow.log_artifact("confusion_matrix.png")


# =============================
# Custom PyFunc Wrapper (Fix dtype issue)
# =============================
class WrappedModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = model
        self.model.eval()

    def predict(self, context, model_input):
        import torch

        input_tensor = torch.tensor(model_input.values, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs.numpy()


mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=WrappedModel(),
    registered_model_name="MultiLabelNLPModel",
)

print("Training complete.")
