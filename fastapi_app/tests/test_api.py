import os
import sys

# Add fastapi_app directory to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_login_success():
    response = client.post("/login", json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    assert "token" in response.json()


def test_login_failure():
    response = client.post("/login", json={"username": "wrong", "password": "wrong"})
    assert response.status_code == 401


def test_predict_without_token():
    response = client.post("/predict", json={"text": "There is pleural fluid."})
    assert response.status_code in [401, 403]


def test_predict_with_token():
    login_response = client.post(
        "/login", json={"username": "admin", "password": "admin"}
    )
    token = login_response.json()["token"]

    headers = {"Authorization": f"Bearer {token}"}

    response = client.post(
        "/predict",
        json={"text": "There is pleural fluid and cardiomegaly."},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
