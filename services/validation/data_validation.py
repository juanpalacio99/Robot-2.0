"""
data_validation.py

Propósito:
    Validar esquemas y rangos en DataFrames usando Great Expectations y Evidently.
    Define dos funciones principales:
      • run_great_expectations: ejecuta validaciones de GE (p. ej., que columnas no estén vacías, tipos correctos).
      • run_evidently_report: genera un reporte de validación con Evidently y lo guarda (HTML o JSON).

Dependencias:
    - great_expectations>=0.15
    - evidently>=0.2
    - pandas

Ejemplo de uso:
    1. Ejecutar validación de GE:
       validation_result = run_great_expectations(df, expectation_suite_path)
    2. Generar reporte Evidently:
       report_path = run_evidently_report(df, ref_df, output_dir)
"""

import os
import pandas as pd
import great_expectations as ge
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab

def run_great_expectations(df: pd.DataFrame, expectation_suite_path: str) -> dict:
    """
    Ejecuta validaciones de Great Expectations usando la definición en 'expectation_suite_path'.
    Retorna el resultado del batch validation.
    """
    # Cargar DataFrame en GE
    context = ge.DataContext()  # Asume que la configuración de GE ya existe en el proyecto
    batch_kwargs = {"dataset": df}
    suite_name = os.path.splitext(os.path.basename(expectation_suite_path))[0]
    suite = context.get_expectation_suite(suite_name)

    validator = context.get_validator(
        batch_kwargs=batch_kwargs,
        expectation_suite_name=suite_name
    )

    # Correr validaciones
    results = validator.validate()
    return results

def run_evidently_report(current_df: pd.DataFrame,
                         reference_df: pd.DataFrame,
                         output_dir: str) -> str:
    """
    Genera un reporte de Data Drift con Evidently comparando current_df vs reference_df.
    Guarda en 'output_dir/report.html' y retorna la ruta al archivo generado.
    """
    os.makedirs(output_dir, exist_ok=True)
    dashboard = Dashboard(tabs=[DataDriftTab(), ClassificationPerformanceTab()])
    dashboard.calculate(current_df=current_df, reference_data=reference_df)

    report_path = os.path.join(output_dir, "evidently_report.html")
    dashboard.save(report_path)
    return report_path
