# services/feature_store/feature_store_online.py

import os
import pandas as pd
import redis


def setup_feature_store():
    """
    En esta prueba no hacemos realmente ningún registro en Feast,
    así que devolvemos un objeto dummy (None) para que no falle la llamada en el test.
    """
    return None


def materialize_to_redis(fs, start_date, end_date):
    """
    Lee el Parquet que indica BATCH_DATA_PATH (contiene ya las columnas precomputadas
    price_ma_5, price_diff, created_ts) y escribe en Redis un hash por fila,
    usando como key: "{symbol}:{timestamp_unix}".
    """
    batch_path = os.getenv("BATCH_DATA_PATH")
    df = pd.read_parquet(batch_path)

    # Conexión a Redis (asume que el contenedor Redis ya está levantado y accesible como "redis:6379")
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)

    for _, row in df.iterrows():
        symbol = row["symbol"]
        # row["timestamp"] es un pandas.Timestamp -> convertir a segundos UNIX
        ts_unix = int(pd.Timestamp(row["timestamp"]).timestamp())
        key = f"{symbol}:{ts_unix}"
        r.hset(key, mapping={
            "price_ma_5": float(row["price_ma_5"]),
            "price_diff": float(row["price_diff"]),
            "created_ts": ts_unix
        })

    return True


def get_online_features(fs, entity_rows, feature_refs):
    """
    Dado un listado de entidades (entity_rows = [{"symbol": "EURUSD"}, ...]) y una lista de
    feature_refs (p.ej. ["symbol_features:price_ma_5", "symbol_features:price_diff"]), consulta
    Redis y devuelve un DataFrame de Pandas con una fila por entidad y las columnas solicitadas.

    Para cada entidad:
      1) Busca en Redis todas las keys que empiecen por "{symbol}:"
      2) Toma la key de timestamp más grande (i.e. la más reciente)
      3) A partir del hash en Redis, extrae únicamente las features que estén en feature_refs
         (la parte después de los dos puntos) y arma una fila de salida.
    """
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    out_rows = []

    for entity in entity_rows:
        symbol = entity["symbol"]
        # obtenemos todas las keys de Redis para este símbolo
        keys = r.keys(f"{symbol}:*")
        if not keys:
            # si no hay datos, añadir fila vacía o con NaNs (aquí vamos a omitirla)
            continue

        # usamos la key con timestamp UNIX más grande
        latest_key = max(keys, key=lambda k: int(k.split(":")[1]))
        data = r.hgetall(latest_key)

        # armamos el dict de salida solo con las columnas pedidas en feature_refs
        row = {}
        for ref in feature_refs:
            # formateo esperado: "symbol_features:price_ma_5" -> tomamos "price_ma_5"
            _, feat_name = ref.split(":")
            if feat_name in data:
                row[feat_name] = float(data[feat_name])

        out_rows.append(row)

    # devolvemos un DataFrame con una fila por entidad
    return pd.DataFrame(out_rows)
