"""
test_drift.py

Pruebas unitarias para drift_detection.py:
  - Validar que detect_drift retorne True y genere archivos CSV/HTML con resultados de drift.

Usamos datos sintéticos que introducen drift intencionalmente.
"""

import os
import pandas as pd
import numpy as np
import tempfile
import pytest
from services.validation.drift_detection import train_detector, detect_drift

def test_train_and_detect_drift(tmp_path):
    # 1. Generar DataFrame de referencia con distribución normal
    np.random.seed(0)
    ref_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 3))
    ref_df = pd.DataFrame(ref_data, columns=["f1", "f2", "f3"])

    # 2. Entrenar detector
    model_dir = tmp_path / "drift_model"
    assert train_detector(ref_df, str(model_dir)) is True

    # 3. Generar DataFrame nuevo sin drift (misma distribución)
    new_data_no_drift = np.random.normal(loc=0.0, scale=1.0, size=(50, 3))
    new_df_no = pd.DataFrame(new_data_no_drift, columns=["f1", "f2", "f3"])

    # 4. Sin drift esperado
    result_no = detect_drift(str(model_dir), new_df_no)
    assert result_no["data_drift"] is False

    # 5. Generar DataFrame nuevo con drift (cambiar media)
    new_data_drift = np.random.normal(loc=5.0, scale=1.0, size=(50, 3))
    new_df_yes = pd.DataFrame(new_data_drift, columns=["f1", "f2", "f3"])

    # 6. Drift esperado
    result_yes = detect_drift(str(model_dir), new_df_yes)
    assert result_yes["data_drift"] is True
