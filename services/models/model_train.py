"""
model_train.py

Propósito:
    Entrenar un modelo de regresión lineal para predecir el precio futuro a partir de features
    batch (price_ma_5, price_diff) y guardar el modelo entrenado en disco.

Dependencias:
    - pandas>=1.0
    - scikit-learn>=1.0
    - joblib>=1.0

Ejemplo de uso:
    1. Crear un Parquet con columnas ["price", "price_ma_5", "price_diff"].
    2. Ejecutar:
         python services/forecast/model_train.py --input-parquet <ruta_entrada> --model-path <ruta_salida>
"""

import argparse
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train_model(input_parquet_path: str, model_output_path: str) -> bool:
    """
    Lee un Parquet con columnas ["price", "price_ma_5", "price_diff"], entrena un modelo
    de regresión lineal para predecir 'price' a partir de ["price_ma_5", "price_diff"], 
    y guarda el modelo entrenado en `model_output_path` usando joblib.

    Args:
        input_parquet_path (str): Ruta al archivo Parquet de entrenamiento.
        model_output_path   (str): Ruta donde se almacenará el modelo entrenado (*.joblib).

    Returns:
        bool: True si el proceso se completa exitosamente.
    """
    # Verificar que el archivo de entrada existe
    if not os.path.exists(input_parquet_path):
        raise FileNotFoundError(f"El archivo de entrenamiento no existe: {input_parquet_path}")

    # Leer DataFrame desde Parquet
    df = pd.read_parquet(input_parquet_path)

    # Verificar que estén las columnas requeridas
    required_cols = {"price", "price_ma_5", "price_diff"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")

    # Separar features (X) y target (y)
    X = df[["price_ma_5", "price_diff"]].values
    y = df["price"].values

    # Entrenar regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Asegurarse de que la carpeta de salida exista
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Guardar modelo con joblib
    joblib.dump(model, model_output_path)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo de regresión lineal y guarda en disco.")
    parser.add_argument(
        "--input-parquet",
        type=str,
        required=True,
        help="Ruta al Parquet de entrenamiento con columnas ['price','price_ma_5','price_diff']"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Ruta donde se guardará el modelo entrenado (*.joblib)"
    )
    args = parser.parse_args()

    success = train_model(args.input_parquet, args.model_path)
    if success:
        print(f"Modelo entrenado y guardado en: {args.model_path}")


