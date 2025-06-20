# tests/test_mlops.py
import subprocess
import requests
import pytest

MLFLOW_URI = "http://localhost:5000"

def run_step(step):
    # Asume que MLflow y W&B están corriendo (pueden ser contenedores mock)
    cmd = ["python", "mlops/mlflow_tracking.py", step]
    subprocess.run(cmd, check=True)

def test_mlops_pipeline(tmp_path):
    # Ejecutar cada paso
    for step in ["preprocess", "train", "evaluate", "register"]:
        run_step(step)
    # Verificar que MLflow Tracking Server responde
    r = requests.get(f"{MLFLOW_URI}/api/2.0/mlflow/experiments/list")
    assert r.status_code == 200
