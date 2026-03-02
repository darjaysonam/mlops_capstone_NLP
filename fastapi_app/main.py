import os
import sys
from datetime import datetime, timedelta
from typing import List

import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# -------------------------------------------------
# Add project root to path
# -------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.nlp.inference import ModelService

# -------------------------------------------------
# App Initialization
# -------------------------------------------------

app = FastAPI(
    title="Radiology Disease Prediction API",
    description="Multi-label NLP Disease Prediction using FastAPI",
    version="1.0",
)

model_service = ModelService()
SECRET_KEY = "supersecretkey"

security = HTTPBearer()

# -------------------------------------------------
# Rate Limiting
# -------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"status": "error", "message": "Rate limit exceeded"},
    )


# -------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    status: str
    token: str


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=5)


class PredictionResult(BaseModel):
    label: str
    probability: float
    predicted: bool


class PredictionResponse(BaseModel):
    status: str
    predictions: List[PredictionResult]


# -------------------------------------------------
# JWT Dependency
# -------------------------------------------------


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# -------------------------------------------------
# Background Task
# -------------------------------------------------


async def log_prediction(text: str):
    print(f"[LOG] Prediction requested at {datetime.utcnow()} | Length: {len(text)}")


# -------------------------------------------------
# Routes
# -------------------------------------------------


@app.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest):

    if data.username != "admin" or data.password != "admin":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = jwt.encode(
        {"user": data.username, "exp": datetime.utcnow() + timedelta(minutes=60)},
        SECRET_KEY,
        algorithm="HS256",
    )

    return {"status": "success", "token": token}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
async def predict(
    request: Request,
    data: PredictionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_token),
):

    background_tasks.add_task(log_prediction, data.text)

    predictions = model_service.predict(data.text)

    return {"status": "success", "predictions": predictions}
