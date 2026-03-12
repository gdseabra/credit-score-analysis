"""
Fixtures compartilhadas entre todos os testes.

Usamos dados sintéticos mínimos — sem dependência dos CSVs reais —
para que os testes rodem em qualquer ambiente (CI/CD, máquina nova, etc.).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import ANOMALY_DAYS_EMPLOYED


@pytest.fixture
def df_minimo() -> pd.DataFrame:
    """DataFrame mínimo com as colunas obrigatórias para AnomalyHandler e DomainFeatureBuilder.

    Contém 10 linhas com valores normais e 2 anomalias em DAYS_EMPLOYED.
    """
    return pd.DataFrame({
        "DAYS_EMPLOYED": [
            ANOMALY_DAYS_EMPLOYED,  # anomalia
            ANOMALY_DAYS_EMPLOYED,  # anomalia
            -1000, -500, -200, -800, -1500, -300, -600, -900,
        ],
        "DAYS_BIRTH": [-12000, -15000, -10000, -8000, -20000, -14000, -11000, -9000, -13000, -16000],
        "AMT_CREDIT": [500000, 300000, 200000, 150000, 800000, 450000, 350000, 250000, 600000, 400000],
        "AMT_INCOME_TOTAL": [100000, 80000, 60000, 50000, 150000, 90000, 70000, 55000, 110000, 85000],
        "AMT_ANNUITY": [25000, 15000, 10000, 8000, 40000, 22000, 17000, 12000, 30000, 20000],
        "CNT_FAM_MEMBERS": [2, 3, 1, 4, 2, 3, 2, 1, 3, 4],
    })


@pytest.fixture
def df_com_target(df_minimo) -> pd.DataFrame:
    """DataFrame mínimo com coluna TARGET para testes de análise."""
    df = df_minimo.copy()
    df["TARGET"] = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    df["CODE_GENDER"] = ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"]
    df["NAME_INCOME_TYPE"] = ["Working", "Commercial", "Working", "Pensioner",
                               "Working", "Commercial", "Working", "Pensioner",
                               "Working", "Commercial"]
    return df


@pytest.fixture
def df_com_anomalia_isolada() -> pd.DataFrame:
    """DataFrame com apenas a anomalia de DAYS_EMPLOYED — para testes unitários focados."""
    return pd.DataFrame({
        "DAYS_EMPLOYED": [ANOMALY_DAYS_EMPLOYED, -500, -1000],
        "DAYS_BIRTH": [-10000, -12000, -15000],
        "AMT_CREDIT": [200000, 300000, 400000],
        "AMT_INCOME_TOTAL": [80000, 100000, 120000],
        "AMT_ANNUITY": [10000, 15000, 20000],
        "CNT_FAM_MEMBERS": [2, 3, 4],
    })
