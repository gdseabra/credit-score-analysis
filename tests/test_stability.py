"""Testes do módulo de estabilidade populacional (PSI)."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.stability import (
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    StabilityLevel,
    categorical_stability_index,
    classify_stability,
    population_stability_index,
    stability_report,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_psi_zero_para_distribuicoes_identicas(rng):
    """Mesma distribuição em referência e atual → PSI praticamente nulo."""
    amostra = rng.normal(size=10_000)
    psi = population_stability_index(amostra, amostra)
    assert psi < 1e-6


def test_psi_alto_para_deslocamento_forte(rng):
    """Deslocamento grande de média → PSI acima do limiar significativo."""
    referencia = rng.normal(loc=0.0, size=10_000)
    atual = rng.normal(loc=3.0, size=10_000)
    psi = population_stability_index(referencia, atual)
    assert psi > PSI_SIGNIFICANT_THRESHOLD


def test_psi_cresce_monotonicamente_com_o_shift(rng):
    """Quanto maior o deslocamento, maior o PSI."""
    referencia = rng.normal(loc=0.0, size=10_000)
    psi_pequeno = population_stability_index(referencia, rng.normal(loc=0.3, size=10_000))
    psi_grande = population_stability_index(referencia, rng.normal(loc=1.5, size=10_000))
    assert psi_grande > psi_pequeno


def test_psi_lida_com_valores_fora_do_intervalo_de_referencia(rng):
    """Valores atuais além do range visto na referência não quebram o cálculo."""
    referencia = rng.uniform(0.0, 1.0, size=1_000)
    atual = rng.uniform(5.0, 6.0, size=1_000)
    psi = population_stability_index(referencia, atual)
    assert np.isfinite(psi) and psi > PSI_SIGNIFICANT_THRESHOLD


def test_psi_ignora_nan(rng):
    """NaN são descartados sem afetar o resultado de distribuições idênticas."""
    base = rng.normal(size=1_000)
    com_nan = np.concatenate([base, [np.nan] * 50])
    psi = population_stability_index(base, com_nan)
    assert psi < 1e-6


def test_psi_levanta_erro_para_base_vazia():
    """Base vazia após remover NaN → ValueError explícito."""
    with pytest.raises(ValueError):
        population_stability_index([np.nan, np.nan], [1.0, 2.0, 3.0])


def test_categorical_psi_zero_para_mesma_distribuicao():
    referencia = pd.Series(["A", "B", "C"] * 100)
    psi = categorical_stability_index(referencia, referencia.copy())
    assert psi < 1e-9


def test_categorical_psi_detecta_categoria_nova():
    """Categoria ausente na referência gera PSI positivo relevante."""
    referencia = pd.Series(["A"] * 90 + ["B"] * 10)
    atual = pd.Series(["A"] * 50 + ["C"] * 50)
    psi = categorical_stability_index(referencia, atual)
    assert psi > PSI_SIGNIFICANT_THRESHOLD


@pytest.mark.parametrize(
    "psi, esperado",
    [
        (0.0, StabilityLevel.STABLE),
        (PSI_MODERATE_THRESHOLD - 1e-9, StabilityLevel.STABLE),
        (PSI_MODERATE_THRESHOLD, StabilityLevel.MODERATE),
        (PSI_SIGNIFICANT_THRESHOLD - 1e-9, StabilityLevel.MODERATE),
        (PSI_SIGNIFICANT_THRESHOLD, StabilityLevel.SIGNIFICANT),
        (1.0, StabilityLevel.SIGNIFICANT),
    ],
)
def test_classify_stability_nos_limiares(psi, esperado):
    """Classificação correta exatamente nas bordas dos limiares."""
    assert classify_stability(psi) == esperado


def test_stability_report_ordena_por_psi_decrescente(rng):
    """O relatório coloca as features mais instáveis no topo."""
    referencia = pd.DataFrame({
        "estavel": rng.normal(loc=0.0, size=2_000),
        "instavel": rng.normal(loc=0.0, size=2_000),
        "categoria": pd.Series(["X", "Y"] * 1_000),
    })
    atual = pd.DataFrame({
        "estavel": rng.normal(loc=0.0, size=2_000),
        "instavel": rng.normal(loc=2.5, size=2_000),
        "categoria": pd.Series(["X", "Y"] * 1_000),
    })

    report = stability_report(referencia, atual)

    assert list(report.columns) == ["feature", "psi", "nivel"]
    assert report.iloc[0]["feature"] == "instavel"
    assert report.iloc[0]["psi"] > report.iloc[-1]["psi"]
