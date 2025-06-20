"""
model_predict.py

Propósito:
    Cargar un modelo entrenado (joblib) y exponer una función para predecir el precio
    futuro a partir de un diccionario de features {"price_ma_5": float, "price_diff": float}.

Dependencias:
    - numpy>=1.20
    - scikit-learn>=1.0
    - joblib>=1.0

Ejemplo de uso:
    1. Asegurarse de tener un modelo entrenado en '*.joblib'.
    2. Desde Python:
         from services.forecast.model_predict import predict_future_price
         features = {"price_ma_5": 4.0, "price_diff": 0.5}
         pred = predict_future_price("ruta/al/modelo.joblib", features)
         print(pred)
"""

import os

import numpy as np
import joblib


def predict_future_price(model_path: str, features: dict) -> float:
    """
    Carga un modelo entrenado de regresión lineal desde `model_path` y predice
    el precio futuro usando los valores en `features`.

    Args:
        model_path (str): Ruta al archivo del modelo (*.joblib).
        features   (dict): Diccionario con llaves:
                               - "price_ma_5": float
                               - "price_diff": float

    Returns:
        float: Precio predicho por el modelo.
    """
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    # Cargar modelo
    model = joblib.load(model_path)

    # Verificar que las keys requeridas estén presentes
    if not {"price_ma_5", "price_diff"}.issubset(features.keys()):
        missing = {"price_ma_5", "price_diff"} - set(features.keys())
        raise ValueError(f"Faltan llaves requeridas en features: {missing}")

    # Construir array de entrada para predict
    X = np.array([[features["price_ma_5"], features["price_diff"]]], dtype=float)

    # Predecir y retornar valor escalar
    pred = model.predict(X)
    return float(pred[0])
