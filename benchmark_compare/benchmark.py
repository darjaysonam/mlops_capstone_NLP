import requests
import time
import numpy as np

# -------------------------
# Config
# -------------------------
MLFLOW_URL = "http://127.0.0.1:5001/invocations"
FASTAPI_URL = "http://127.0.0.1:8000/predict"  # change if needed

N_REQUESTS = 100

payload_mlflow = {
    "dataframe_split": {
        "columns": [f"f{i}" for i in range(1, 21)],
        "data": [np.random.rand(20).tolist()],
    }
}

payload_fastapi = {"inputs": np.random.rand(20).tolist()}


# -------------------------
# Benchmark Function
# -------------------------
def benchmark(url, payload, name):
    start = time.time()
    for _ in range(N_REQUESTS):
        requests.post(url, json=payload)
    end = time.time()
    avg_latency = (end - start) / N_REQUESTS
    print(f"{name} Average Latency: {avg_latency:.6f} seconds")


# -------------------------
# Run Benchmarks
# -------------------------
print("Running benchmark with", N_REQUESTS, "requests...\n")

benchmark(MLFLOW_URL, payload_mlflow, "MLflow")
benchmark(FASTAPI_URL, payload_fastapi, "FastAPI")
