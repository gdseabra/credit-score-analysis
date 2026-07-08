"""
Estabilidade populacional (Population Stability Index — PSI).

O PSI mede o quanto a distribuição de uma variável mudou entre uma população
de *referência* (tipicamente a base de desenvolvimento do modelo) e uma
população *atual* (a que está entrando em produção). É o instrumento padrão da
indústria de crédito para detectar *data drift* antes que ele degrade o modelo.

Duas aplicações distintas do mesmo cálculo:
    - PSI de *feature*  — a distribuição de uma variável de entrada se moveu?
    - PSI de *score*    — a distribuição das probabilidades preditas se moveu?
      (também chamado de Characteristic/Score Stability Index)

Regra de bolso da indústria para interpretar o valor:
    PSI < 0.10          → população estável
    0.10 ≤ PSI < 0.25   → mudança moderada (monitorar)
    PSI ≥ 0.25          → mudança significativa (investigar / recalibrar)
"""

from enum import Enum

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------

PSI_MODERATE_THRESHOLD: float = 0.10
"""Limiar acima do qual a mudança populacional é considerada moderada."""

PSI_SIGNIFICANT_THRESHOLD: float = 0.25
"""Limiar acima do qual a mudança populacional é considerada significativa."""

DEFAULT_N_BINS: int = 10
"""Número de faixas (decis) usado para discretizar variáveis contínuas."""

EPSILON: float = 1e-6
"""Piso aplicado às proporções para evitar log(0) e divisão por zero em bins vazios."""


class StabilityLevel(str, Enum):
    """Classificação qualitativa de um valor de PSI."""

    STABLE = "estável"
    MODERATE = "mudança moderada"
    SIGNIFICANT = "mudança significativa"


def classify_stability(psi: float) -> StabilityLevel:
    """Traduz um valor numérico de PSI na sua classificação qualitativa.

    Args:
        psi: Valor do Population Stability Index (>= 0).

    Returns:
        StabilityLevel correspondente segundo os limiares da indústria.
    """
    if psi < PSI_MODERATE_THRESHOLD:
        return StabilityLevel.STABLE
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return StabilityLevel.MODERATE
    return StabilityLevel.SIGNIFICANT


def _psi_from_proportions(ref_prop: np.ndarray, cur_prop: np.ndarray) -> float:
    """Calcula o PSI a partir das proporções por faixa de duas populações.

    PSI = Σ (cur% − ref%) · ln(cur% / ref%)

    As proporções recebem um piso EPSILON para que faixas vazias não gerem
    log(0) nem divisão por zero.
    """
    ref = np.clip(ref_prop, EPSILON, None)
    cur = np.clip(cur_prop, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _quantile_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Define as bordas das faixas por quantis (decis) da população de referência.

    As bordas extremas são abertas (±infinito) para que qualquer valor da
    população atual — inclusive fora do intervalo visto na referência — caia
    em alguma faixa. Quantis duplicados (variáveis muito concentradas) são
    colapsados via np.unique.
    """
    quantis = np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantis)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Calcula o PSI de uma variável *contínua* (feature ou score).

    As faixas são definidas pelos decis da população de referência; a população
    atual é distribuída nessas mesmas faixas. Valores ausentes (NaN) são
    descartados em ambas as populações antes do cálculo — a estabilidade da
    taxa de nulos deve ser monitorada separadamente.

    Args:
        reference: Distribuição de referência (base de desenvolvimento).
        current: Distribuição atual (base a monitorar).
        n_bins: Número de faixas (decis por padrão).

    Returns:
        Valor do PSI (>= 0). Quanto maior, maior o deslocamento populacional.

    Raises:
        ValueError: Se qualquer das populações ficar vazia após remover NaN.
    """
    ref = pd.Series(reference).dropna().to_numpy()
    cur = pd.Series(current).dropna().to_numpy()

    if ref.size == 0 or cur.size == 0:
        raise ValueError("População de referência ou atual vazia após remover NaN.")

    edges = _quantile_edges(ref, n_bins)
    ref_prop = np.histogram(ref, bins=edges)[0] / ref.size
    cur_prop = np.histogram(cur, bins=edges)[0] / cur.size

    return _psi_from_proportions(ref_prop, cur_prop)


def categorical_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
) -> float:
    """Calcula o PSI de uma variável *categórica*.

    Cada categoria é uma faixa. Categorias presentes em apenas uma das
    populações são tratadas via o piso EPSILON em `_psi_from_proportions`.

    Args:
        reference: Distribuição de referência.
        current: Distribuição atual.

    Returns:
        Valor do PSI (>= 0).
    """
    ref = pd.Series(reference).astype("object")
    cur = pd.Series(current).astype("object")

    categorias = sorted(set(ref.dropna()) | set(cur.dropna()), key=str)
    ref_prop = np.array([(ref == c).mean() for c in categorias])
    cur_prop = np.array([(cur == c).mean() for c in categorias])

    return _psi_from_proportions(ref_prop, cur_prop)


def stability_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = DEFAULT_N_BINS,
) -> pd.DataFrame:
    """Gera um relatório de PSI por coluna entre duas bases.

    Colunas numéricas usam PSI por decis; colunas categóricas usam PSI por
    categoria. Apenas as colunas presentes em ambas as bases são avaliadas.

    Args:
        reference: Base de referência.
        current: Base atual.
        n_bins: Número de faixas para variáveis contínuas.

    Returns:
        DataFrame com colunas ``feature``, ``psi`` e ``nivel``, ordenado do
        maior para o menor PSI (features mais instáveis no topo).
    """
    colunas = [c for c in reference.columns if c in current.columns]

    linhas = []
    for coluna in colunas:
        if pd.api.types.is_numeric_dtype(reference[coluna]):
            psi = population_stability_index(reference[coluna], current[coluna], n_bins)
        else:
            psi = categorical_stability_index(reference[coluna], current[coluna])

        linhas.append({
            "feature": coluna,
            "psi": psi,
            "nivel": classify_stability(psi).value,
        })

    return (
        pd.DataFrame(linhas)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )
