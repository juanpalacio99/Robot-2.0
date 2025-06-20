"""
drift_detection.py

Propósito:
    Detectar drift en datos usando Alibi Detect (versiones >= 0.11).
    Incluye funciones para:
      • Entrenar o cargar un detector de drift.
      • Evaluar nuevos datos contra los datos de referencia y reportar si hay drift.

Dependencias:
    - alibi-detect>=0.11
    - pandas
    - joblib (para persistir el detector)
    - numpy

Ejemplo de uso:
    1. Entrenar detector sobre un DataFrame de referencia:
       train_detector(ref_df, model_dir)
    2. Detectar drift en datos nuevos:
       drift_result = detect_drift(model_dir, new_df)
"""

"""
drift_detection.py

Detecta drift en datos tabulares usando Alibi Detect (>=0.11).
"""

import os
import numpy as np
import pandas as pd
from alibi_detect.cd import TabularDrift
from alibi_detect.utils.saving import save_detector, load_detector

def train_detector(ref_df: pd.DataFrame, model_dir: str) -> bool:
    """
    Entrena un detector de drift sobre ref_df y lo guarda en model_dir/drift_detector.pkl
    """
    os.makedirs(model_dir, exist_ok=True)
    # Solo columnas numéricas
    X_ref = ref_df.select_dtypes(include=[np.number]).values

    # Crear detector de drift
    detector = TabularDrift(
        X_ref,
        p_val=0.05,
        n_features=X_ref.shape[1],
        model_kwargs={'n_estimators': 50, 'max_features': 'auto'}
    )

    # Guardar detector
    save_detector(detector, os.path.join(model_dir, "drift_detector.pkl"))
    return True

def detect_drift(model_dir: str, new_df: pd.DataFrame) -> dict:
    """
    Carga el detector entrenado y evalúa drift sobre new_df.
    Devuelve dict con keys: data_drift (bool), p_val (float), feature_score (list o None).
    """
    detector_path = os.path.join(model_dir, "drift_detector.pkl")
    if not os.path.exists(detector_path):
        raise FileNotFoundError(f"Detector no encontrado en {detector_path}")

    detector = load_detector(detector_path)
    X_new = new_df.select_dtypes(include=[np.number]).values
    preds = detector.predict(X_new)

    return {
        "data_drift": bool(preds["data_drift"]),
        "p_val": float(np.mean(preds["p_val"])),
        "feature_score": preds.get("feature_score", None).tolist() if "feature_score" in preds else None
    }
