# services/forecast/forecast_model.py

"""
forecast_model.py

Propósito:
    - Entrenar un modelo de regresión lineal a partir de un Parquet de entrenamiento.
    - Serializar el modelo a disco.
    - Cargar el modelo para predecir el precio futuro, dado un diccionario de features.

Dependencias:
    - pandas
    - scikit-learn
    - joblib
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train_model(input_parquet: str, model_path: str) -> bool:
    """
    Lee un archivo Parquet con columnas mínimas:
      - price (etiqueta a predecir)
      - price_ma_5, price_diff (features de entrada en este ejemplo)
    Entrena un LinearRegression y guarda el modelo serializado en 'model_path'.

    Parámetros:
      input_parquet: ruta al Parquet de entrenamiento.
      model_path: ruta (incluyendo nombre .joblib) donde se guardará el modelo.

    Retorna:
      True si el entrenamiento y guardado fueron exitosos.
    """
    # Leer el DataFrame
    df = pd.read_parquet(input_parquet)

    # Verificar columnas mínimas
    required_cols = {"price", "price_ma_5", "price_diff"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"El Parquet debe contener al menos las columnas: {required_cols}")

    # Definir X e y
    X = df[["price_ma_5", "price_diff"]].fillna(0)
    y = df["price"]

    # (Opcional) Dividir en train/test; en este módulo solo entrenamos con todo
    # Si se quiere validación, se podría hacer:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = LinearRegression().fit(X_train, y_train)

    model = LinearRegression().fit(X, y)

    # Asegurarnos de que la carpeta existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Serializar el modelo con joblib
    joblib.dump(model, model_path)

    return True

def predict_future_price(model_path: str, features: dict) -> float:
    """
    Carga un modelo entrenado (serializado con joblib) desde 'model_path'
    y predice el precio dado un diccionario de features:
      features debe tener las claves:
        - price_ma_5
        - price_diff

    Retorna la predicción (float).
    """
    # Cargar el modelo desde disco
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en ruta: {model_path}")

    model = joblib.load(model_path)

    # Validar que el diccionario contenga las llaves esperadas
    expected_keys = {"price_ma_5", "price_diff"}
    if not expected_keys.issubset(set(features.keys())):
        raise ValueError(f"El diccionario 'features' debe contener: {expected_keys}")

    # Crear un DataFrame de una sola fila
    X_new = pd.DataFrame(
        {
            "price_ma_5": [features["price_ma_5"]],
            "price_diff": [features["price_diff"]]
        }
    )

    # Hacer la predicción
    y_pred = model.predict(X_new)

    # Devolver el valor escalar
    return float(y_pred[0])
