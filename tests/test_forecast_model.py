# tests/test_forecast_model.py

import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import pytest
from sklearn.linear_model import LinearRegression

from services.forecast.forecast_model import train_model, predict_future_price

def test_train_and_predict(tmp_path):
    """
    - Creamos un DataFrame sintético donde price = 2 * price_ma_5 + 3 * price_diff + 5.
    - Guardamos a Parquet.
    - Entrenamos el modelo.
    - Hacemos una predicción y comprobamos que esté muy cerca del valor real.
    """
    # 1. Construir DataFrame sintético
    np.random.seed(42)
    n = 50
    price_ma_5 = np.random.uniform(1, 10, size=n)
    price_diff = np.random.uniform(-1, 1, size=n)
    price = 2.0 * price_ma_5 + 3.0 * price_diff + 5.0

    df = pd.DataFrame({
        "price": price,
        "price_ma_5": price_ma_5,
        "price_diff": price_diff
    })

    # 2. Guardar DataFrame a Parquet
    input_parquet = tmp_path / "train_data.parquet"
    df.to_parquet(str(input_parquet), index=False)

    # 3. Definir ruta donde guardaremos el modelo
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / "linear_model.joblib")

    # 4. Llamar a train_model
    assert train_model(str(input_parquet), model_path) is True

    # 5. Comprobar que el archivo del modelo existe
    assert os.path.exists(model_path)

    # 6. Hacer predicción para un par de ejemplos
    # Ejemplo A: price_ma_5=4, price_diff=0.5 -> price esperado = 2*4 + 3*0.5 + 5 = 8 + 1.5 + 5 = 14.5
    features_A = {"price_ma_5": 4.0, "price_diff": 0.5}
    pred_A = predict_future_price(model_path, features_A)
    assert pytest.approx(pred_A, rel=1e-2) == 14.5

    # Ejemplo B: price_ma_5=7, price_diff=-0.2 -> price esperado = 2*7 + 3*(-0.2) + 5 = 14 - 0.6 + 5 = 18.4
    features_B = {"price_ma_5": 7.0, "price_diff": -0.2}
    pred_B = predict_future_price(model_path, features_B)
    assert pytest.approx(pred_B, rel=1e-2) == 18.4
