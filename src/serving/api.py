"""
FastAPI Serving Layer for Chest X-ray Model
Includes:
- Image upload
- Prediction
- Prometheus monitoring
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

from src.serving.model_loader import load_model
from src.monitoring.metrics_collector import REQUEST_COUNT

# ───────────────────────────────────────────────
# Initialize FastAPI App
# ───────────────────────────────────────────────

app = FastAPI(
    title="Chest X-ray Diagnosis API",
    description="Predicts disease labels from chest X-ray images",
    version="1.0"
)

# ───────────────────────────────────────────────
# Load Model Once at Startup
# ───────────────────────────────────────────────

model = load_model()

# Set model to evaluation mode
model.eval()

# ───────────────────────────────────────────────
# Image Preprocessing (MUST match training)
# ───────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ───────────────────────────────────────────────
# Health Check Endpoint
# ───────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "API is running"}

# ───────────────────────────────────────────────
# Prediction Endpoint
# ───────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload chest X-ray image and get predictions
    """

    try:
        # Count API requests for Prometheus
        REQUEST_COUNT.inc()

        # Read uploaded file
        contents = await file.read()

        # Convert to image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply preprocessing
        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Disable gradient calculation (inference mode)
        with torch.no_grad():
            outputs = model(image)

        # Convert output tensor to list
        predictions = outputs.tolist()

        return JSONResponse(
            content={
                "success": True,
                "predictions": predictions
            }
        )

    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )