"""
test_feature_store.py

Pruebas unitarias para el Módulo 2: Feature Store híbrido (Batch + Online).

Dependencias:
    - pytest
    - pyspark>=3.3
    - pandas>=1.5
    - feast>=0.20
    - redis>=6.2

Ejemplo de uso:
    1. Inicia Redis:
       docker-compose -f deploy/docker-compose.redis.yml up -d
    2. Ejecuta:
       pytest tests/test_feature_store.py

Según Sección “Módulo 2 – Feature Store híbrido” de Instrucciones.pdf, estas pruebas validan:
  • Que la función batch genere las columnas price_ma_5, price_diff y created_ts correctamente.
  • Que la capa online (Feast + Redis) pueda materializar y recuperar valores (opcional).
"""

import os
import pandas as pd
import pytest
from pyspark.sql import SparkSession

# Importamos la función batch implementada en feature_store_batch.py
from services.feature_store.feature_store_batch import generate_batch_features

# Importamos la capa online (las funciones que acabamos de ajustar):
from services.feature_store.feature_store_online import (
    setup_feature_store,
    materialize_to_redis,
    get_online_features
)

def test_batch_feature_generation(tmp_path):
    """
    Verifica que generate_batch_features:
      1. Lea un Parquet de entrada con columnas [symbol, timestamp, price, volume].
      2. Genere las columnas:
         - price_ma_5: media móvil de precio en ventana de 5 registros.
         - price_diff: diferencia de precio respecto al registro anterior.
         - created_ts: igual al campo timestamp original.
      3. Escriba un Parquet de salida con dichas columnas.
    """

    # 1. Crear SparkSession local (la función batch usará getOrCreate())
    spark = SparkSession.builder \
        .appName("TestFeatureBatch") \
        .master("local[*]") \
        .getOrCreate()

    # 2. Crear DataFrame de ejemplo con columnas: symbol, timestamp, price, volume
    data = [
        ("EURUSD", 1, 1.0, 100),
        ("EURUSD", 2, 2.0, 150),
        ("EURUSD", 3, 3.0, 200),
        ("EURUSD", 4, 4.0, 250),
        ("EURUSD", 5, 5.0, 300),
        ("EURUSD", 6, 6.0, 350),
    ]
    columns = ["symbol", "timestamp", "price", "volume"]
    df_input = spark.createDataFrame(data, columns)

    # 3. Guardar DataFrame de entrada como Parquet en tmp_path
    input_path = tmp_path / "input.parquet"
    df_input.write.parquet(str(input_path))

    # 4. Definir ruta de salida para las features
    output_path = tmp_path / "features.parquet"

    # 5. Ejecutar la función batch
    generate_batch_features(str(input_path), str(output_path))

    # 6. Leer el Parquet resultante y convertir a pandas
    df_features = spark.read.parquet(str(output_path)).toPandas()

    # 7. Verificar que existan las columnas generadas
    assert "price_ma_5" in df_features.columns, "Falta la columna price_ma_5"
    assert "price_diff" in df_features.columns, "Falta la columna price_diff"
    assert "created_ts" in df_features.columns, "Falta la columna created_ts"

    # 8. Verificar valores puntuales:
    #    - price_ma_5 en la fila con timestamp==5 debe ser (1+2+3+4+5)/5 = 3.0
    row_5 = df_features[df_features["timestamp"] == 5].iloc[0]
    assert pytest.approx(row_5["price_ma_5"], rel=1e-3) == 3.0

    #    - price_diff en la fila con timestamp==6 debe ser 6.0 - 5.0 = 1.0
    row_6 = df_features[df_features["timestamp"] == 6].iloc[0]
    assert pytest.approx(row_6["price_diff"], rel=1e-3) == 1.0

    #    - created_ts coincide con el timestamp original en todas las filas
    assert all(df_features["created_ts"] == df_features["timestamp"]), \
        "El campo created_ts no coincide con timestamp original"

    # 9. Cerrar SparkSession al terminar
    spark.stop()


def test_online_feature_store(monkeypatch, tmp_path):
    """
    Verifica la capa online (Feast + Redis):
      • setup_feature_store: inicializa conexiones a Redis y a Feast.
      • materialize_to_redis: carga datos batch a Redis entre fechas.
      • get_online_features: recupera features para entidad específica.
    """

    # 1. Crear repo temporal de Feast (no se usa realmente, solo para la variable de entorno)
    repo_dir = tmp_path / "feat_repo"
    os.makedirs(repo_dir, exist_ok=True)

    # 2. Crear DataFrame de pandas de ejemplo batch con columnas requeridas
    batch_data = [
        {
            "symbol": "EURUSD",
            # Fecha que convertiremos a segundos UNIX dentro de materialize_to_redis
            "timestamp": pd.Timestamp("2025-06-02T00:00:00"),
            "price_ma_5": 3.0,
            "price_diff": 1.0,
            "created_ts": pd.Timestamp("2025-06-02T00:00:00")
        },
    ]
    batch_df = pd.DataFrame(batch_data)

    # 3. Guardar como Parquet temporal
    batch_path = tmp_path / "batch_data.parquet"
    batch_df.to_parquet(str(batch_path), index=False)

    # 4. Inyectar variables de entorno para que feature_store_online use este Parquet
    monkeypatch.setenv("BATCH_DATA_PATH", str(batch_path))
    monkeypatch.setenv("FEAST_REPO_PATH", str(repo_dir))

    # 5. Inicializar Feast + Redis (setup_feature_store no hace realmente la parte de Redis,
    #    sino que registra la entidad y features en un repo en memoria)
    fs = setup_feature_store()

    # 6. Materializar datos batch a Redis entre fechas (aunque la prueba no los filtra por fecha,
    #    el método materialize_to_redis solo lee todo el Parquet)
    from datetime import datetime, timedelta
    end = datetime(2025, 6, 2)
    start = end - timedelta(days=1)
    # Devuelve True si todo se guardó correctamente en Redis
    assert materialize_to_redis(fs, start_date=start, end_date=end) is True

    # 7. Consultar features online para entidad 'EURUSD'
    entity_rows = [{"symbol": "EURUSD"}]
    feature_refs = ["symbol_features:price_ma_5", "symbol_features:price_diff"]
    df_online = get_online_features(fs, entity_rows, feature_refs)

    # 8. Verificar que existan las columnas y los valores correctos
    assert "price_ma_5" in df_online.columns, "Falta columna price_ma_5 en online"
    assert "price_diff" in df_online.columns, "Falta columna price_diff en online"
    # Como solo había un registro, revisamos la primera fila
    assert df_online.iloc[0]["price_ma_5"] == pytest.approx(3.0, rel=1e-3), "Valor incorrecto price_ma_5 (online)"
    assert df_online.iloc[0]["price_diff"] == pytest.approx(1.0, rel=1e-3), "Valor incorrecto price_diff (online)"
