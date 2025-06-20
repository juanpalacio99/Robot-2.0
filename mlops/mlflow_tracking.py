#!/usr/bin/env python3
"""
mlops/mlflow_tracking.py

Propósito:
    Orquestar un pipeline sencillo de MLOps con MLflow:
      - preprocess: generar y guardar datos de ejemplo
      - train: entrenar un modelo lineal y loggear artefactos/metrics
      - evaluate: evaluar el modelo y loggear métrica de error
      - register: registrar la versión del modelo en el registry de MLflow

Dependencias:
    mlflow>=2.5, scikit-learn, pandas, numpy

Uso:
    python mlops/mlflow_tracking.py [preprocess|train|evaluate|register]

Ejemplo:
    python mlops/mlflow_tracking.py preprocess
"""

import argparse
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "mlops_pipeline"

def preprocess():
    # 1) Generar datos sintéticos
    df = pd.DataFrame({
        "x": np.arange(100),
        "y": 2 * np.arange(100) + np.random.normal(0, 5, 100)
    })
    os.makedirs("mlops/artifacts", exist_ok=True)
    path = "mlops/artifacts/preprocessed.csv"
    df.to_csv(path, index=False)
    mlflow.log_artifact(path)

def train():
    # 2) Cargar datos
    df = pd.read_csv("mlops/artifacts/preprocessed.csv")
    X = df[["x"]]
    y = df["y"]
    # 3) Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    # 4) Loggear
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("train_r2", model.score(X, y))
    # 5) Guardar run_id para evaluate posterior
    run_id = mlflow.active_run().info.run_id
    with open("mlops/artifacts/train_run_id.txt", "w") as f:
        f.write(run_id)

def evaluate():
    # 5) Leer run_id del entrenamiento y traer modelo desde el Tracking Server
    with open("mlops/artifacts/train_run_id.txt", "r") as f:
        train_run_id = f.read().strip()
    model_uri = f"runs:/{train_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    df = pd.read_csv("mlops/artifacts/preprocessed.csv")
    preds = model.predict(df[["x"]])
    mse = mean_squared_error(df["y"], preds)
    mlflow.log_metric("mse", float(mse))

def register():
    # 6) Registrar en el Model Registry
    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    try:
        client.create_registered_model("LinearModel")
    except mlflow.exceptions.RestException:
        pass  # ya existe
    client.create_model_version(
        name="LinearModel",
        source=model_uri,
        run_id=mlflow.active_run().info.run_id
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["preprocess","train","evaluate","register"])
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        if args.step == "preprocess":
            preprocess()
        elif args.step == "train":
            train()
        elif args.step == "evaluate":
            evaluate()
        elif args.step == "register":
            register()

if __name__ == "__main__":
    main()
