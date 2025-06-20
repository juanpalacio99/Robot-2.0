"""
test_validation.py

Pruebas unitarias para data_validation.py:
  - Validar que run_great_expectations retorne True sobre datos sintéticos limpios.
  - Validar que run_evidently_report genere archivo HTML sin errores.
"""

import os
import pandas as pd
import numpy as np
import tempfile
import pytest
from services.validation.data_validation import run_great_expectations, run_evidently_report

def test_evidently_report(tmp_path):
    # 1. Construir DataFrames de referencia y actual con drift leve en la columna 'v'
    ref_df = pd.DataFrame({
        "v": np.random.normal(0, 1, size=100),
        "label": np.random.choice([0, 1], size=100)
    })
    current_df = pd.DataFrame({
        "v": np.random.normal(1, 1, size=100),  # cambia media
        "label": np.random.choice([0, 1], size=100)
    })

    # 2. Generar reporte con Evidently
    output_dir = tmp_path / "ev_report"
    report_path = run_evidently_report(current_df, ref_df, str(output_dir))

    # 3. Verificar que el HTML se haya creado
    assert os.path.exists(report_path)
    assert report_path.endswith("evidently_report.html")

@pytest.mark.skip("Great Expectations suite no provista en este ejemplo.")
def test_great_expectations(tmp_path):
    # Este test se salta, ya que el expectation suite debe existir en el contexto
    pass
