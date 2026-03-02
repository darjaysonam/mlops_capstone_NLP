import os
import sys
import logging
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, render_template
import jwt

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.nlp.inference import ModelService


# -------------------------------------------------
# Flask App Setup
# -------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"

# Load model once at startup
model_service = ModelService()


# -------------------------------------------------
# Logging Setup
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------------------------
# JWT Authentication Decorator
# -------------------------------------------------

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({
                "status": "error",
                "message": "Token is missing"
            }), 401

        try:
            token = auth_header.split(" ")[1]
            jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        except Exception:
            return jsonify({
                "status": "error",
                "message": "Invalid or expired token"
            }), 401

        return f(*args, **kwargs)

    return decorated


# -------------------------------------------------
# Login Route (POST)
# -------------------------------------------------

@app.route("/login", methods=["POST"])
def login():

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request must be JSON"
        }), 400

    data = request.get_json()

    if data.get("username") != "admin" or data.get("password") != "admin":
        return jsonify({
            "status": "error",
            "message": "Invalid credentials"
        }), 401

    token = jwt.encode({
        "user": data["username"],
        "exp": datetime.utcnow() + timedelta(minutes=60)
    }, app.config["SECRET_KEY"], algorithm="HS256")

    return jsonify({
        "status": "success",
        "token": token
    }), 200


# -------------------------------------------------
# REST Prediction Endpoint (Protected)
# -------------------------------------------------

@app.route("/predict", methods=["POST"])
@token_required
def predict():

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request must be JSON"
        }), 400

    data = request.get_json()
    text = data.get("text")

    if not text or not isinstance(text, str):
        return jsonify({
            "status": "error",
            "message": "Invalid input text"
        }), 400

    logging.info("Prediction request received")

    predictions = model_service.predict(text)

    return jsonify({
        "status": "success",
        "predictions": predictions
    }), 200


# -------------------------------------------------
# HTML Frontend (Jinja2)
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    predictions = None

    if request.method == "POST":
        text = request.form.get("text")

        if text:
            predictions = model_service.predict(text)

    return render_template("index.html", predictions=predictions)


# -------------------------------------------------
# Run App
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)