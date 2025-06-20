"""
feature_store_batch.py

Propósito:
    Generar features de manera batch usando PySpark. 
    Lee un Parquet de entrada con columnas mínimas [symbol, timestamp, price, volume], 
    calcula las siguientes columnas de features:
      - price_ma_5: media móvil de 'price' para las últimas 5 filas (por ventana ordenada por timestamp).
      - price_diff: diferencia de 'price' respecto al valor anterior (lag 1).
      - created_ts: timestamp original (columna 'timestamp').

Dependencias:
    - Python 3.10
    - pyspark>=3.3, <4.0
    - Java 8 o Java 11

Ejemplo de uso (desde línea de comandos):
    python services/feature_store/feature_store_batch.py \\
        --input-path /ruta/al/parquet_entrada.parquet \\
        --output-path /ruta/al/parquet_salida.parquet

Bloque de comandos para pruebas:
    pytest tests/test_feature_store.py

Según Sección “Módulo 2 – Feature Store híbrido” de Instrucciones.pdf, este script debe crear
un SparkSession, leer los datos de entrada, generar las columnas de features indicadas y escribir
el resultado en Parquet, **sin llamar a spark.stop()** (el test controla el ciclo de vida del contexto). :contentReference[oaicite:10]{index=10}
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg, lag

def generate_batch_features(input_path: str, output_path: str):
    """
    Lee datos de entrada desde 'input_path' (formato Parquet), calcula las features y
    escribe el DataFrame resultante en 'output_path' como Parquet.

    Args:
        input_path (str): Ruta al Parquet de entrada con columnas [symbol, timestamp, price, volume].
        output_path (str): Ruta donde se escribirá el Parquet resultante con las nuevas columnas.
    """
    # 1. Crear o recuperar un SparkSession
    spark = SparkSession.builder \
        .appName("FeatureStoreBatch") \
        .master("local[*]") \
        .getOrCreate()

    # 2. Leer DataFrame de entrada (se asumen columnas: symbol, timestamp, price, volume)
    df = spark.read.parquet(input_path)

    # 3. Definir ventana para media móvil de 5 registros (ordenada por timestamp)
    window_spec = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-4, 0)

    # 4. Calcular price_ma_5: media móvil de precio en la ventana de 5 filas
    df_features = df.withColumn("price_ma_5", avg(col("price")).over(window_spec))

    # 5. Calcular price_diff: diferencia de precio respecto al valor anterior (lag 1)
    df_features = df_features.withColumn(
        "price_diff",
        col("price") - lag(col("price"), 1).over(Window.partitionBy("symbol").orderBy("timestamp"))
    )

    # 6. Mantener created_ts igual al timestamp original
    df_features = df_features.withColumn("created_ts", col("timestamp"))

    # 7. Escribir Parquet de salida (modo overwrite)
    df_features.write.mode("overwrite").parquet(output_path)

    # NOTA: NO llamar a spark.stop() aquí. La prueba (pytest) controla el cierre de SparkContext.
    # Si se llama a spark.stop() antes de tiempo, surge un "Cannot call methods on a stopped SparkContext".

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador batch de features usando PySpark")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Ruta de entrada de datos raw en Parquet (columnas: symbol, timestamp, price, volume)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Ruta donde guardar el Parquet resultante con las nuevas columnas de features"
    )
    args = parser.parse_args()
    generate_batch_features(args.input_path, args.output_path)

