"""
Testes de hipóteses estatísticos para o dataset Home Credit Default Risk.

Cada função testa uma hipótese sobre a relação entre uma variável e TARGET,
retornando estatística, p-valor e conclusão legível.

Convenção: H0 (hipótese nula) = não há diferença entre os grupos.
           Rejeitamos H0 quando p-valor < alpha (padrão: 0.05).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TARGET_COL = "TARGET"
ALPHA_PADRAO = 0.05


@dataclass
class ResultadoTeste:
    """Resultado estruturado de um teste de hipótese.

    Attributes:
        nome_teste: Nome do teste estatístico utilizado.
        variavel: Coluna testada.
        estatistica: Valor da estatística do teste.
        p_valor: P-valor calculado.
        alpha: Nível de significância adotado.
        rejeita_h0: True se p-valor < alpha.
        conclusao: Interpretação em linguagem natural.
    """

    nome_teste: str
    variavel: str
    estatistica: float
    p_valor: float
    alpha: float
    rejeita_h0: bool
    conclusao: str

    def __str__(self) -> str:
        status = "REJEITA H0" if self.rejeita_h0 else "NAO REJEITA H0"
        return (
            f"[{self.nome_teste}] {self.variavel}\n"
            f"  Estatística = {self.estatistica:.4f} | p-valor = {self.p_valor:.4e}\n"
            f"  {status} (alpha={self.alpha}) → {self.conclusao}"
        )


def teste_chi2(
    df: pd.DataFrame, coluna: str, alpha: float = ALPHA_PADRAO
) -> ResultadoTeste:
    """Teste Qui-Quadrado de independência entre uma variável categórica e TARGET.

    H0: A variável ``coluna`` é independente do TARGET (sem associação).
    H1: Existe associação estatisticamente significativa.

    Uso adequado para variáveis como NAME_INCOME_TYPE, CODE_GENDER, NAME_EDUCATION_TYPE.

    Args:
        df: DataFrame com as colunas ``coluna`` e ``TARGET``.
        coluna: Nome da variável categórica.
        alpha: Nível de significância (padrão: 0.05).

    Returns:
        ResultadoTeste com estatística chi2, p-valor e conclusão.

    Examples:
        >>> resultado = teste_chi2(df_train, "CODE_GENDER")
        >>> print(resultado)
    """
    validos = df[[coluna, TARGET_COL]].dropna()
    tabela = pd.crosstab(validos[coluna], validos[TARGET_COL])
    chi2, p_valor, _, _ = stats.chi2_contingency(tabela)
    rejeita = p_valor < alpha

    if rejeita:
        conclusao = (
            f"'{coluna}' tem associação significativa com inadimplência "
            f"(p={p_valor:.2e} < {alpha})."
        )
    else:
        conclusao = (
            f"'{coluna}' não tem associação significativa com inadimplência "
            f"(p={p_valor:.2e} >= {alpha})."
        )

    return ResultadoTeste(
        nome_teste="Qui-Quadrado",
        variavel=coluna,
        estatistica=round(chi2, 4),
        p_valor=round(p_valor, 6),
        alpha=alpha,
        rejeita_h0=rejeita,
        conclusao=conclusao,
    )


def teste_mann_whitney(
    df: pd.DataFrame, coluna: str, alpha: float = ALPHA_PADRAO
) -> ResultadoTeste:
    """Teste Mann-Whitney U para comparar uma variável contínua entre adimplentes e inadimplentes.

    H0: A distribuição de ``coluna`` é igual nos dois grupos (TARGET=0 e TARGET=1).
    H1: As distribuições são diferentes.

    Preferível ao t-test quando a distribuição é assimétrica (ex: renda, crédito).

    Args:
        df: DataFrame com as colunas ``coluna`` e ``TARGET``.
        coluna: Nome da variável contínua.
        alpha: Nível de significância (padrão: 0.05).

    Returns:
        ResultadoTeste com estatística U, p-valor e conclusão.

    Examples:
        >>> resultado = teste_mann_whitney(df_train, "AMT_INCOME_TOTAL")
        >>> print(resultado)
    """
    validos = df[[coluna, TARGET_COL]].dropna()
    grupo_0 = validos.loc[validos[TARGET_COL] == 0, coluna]
    grupo_1 = validos.loc[validos[TARGET_COL] == 1, coluna]

    stat, p_valor = stats.mannwhitneyu(grupo_0, grupo_1, alternative="two-sided")
    rejeita = p_valor < alpha

    mediana_0 = grupo_0.median()
    mediana_1 = grupo_1.median()

    if rejeita:
        conclusao = (
            f"'{coluna}' difere significativamente entre grupos "
            f"(mediana pagador={mediana_0:.2f}, inadimplente={mediana_1:.2f}; "
            f"p={p_valor:.2e} < {alpha})."
        )
    else:
        conclusao = (
            f"'{coluna}' não difere significativamente entre grupos "
            f"(p={p_valor:.2e} >= {alpha})."
        )

    return ResultadoTeste(
        nome_teste="Mann-Whitney U",
        variavel=coluna,
        estatistica=round(float(stat), 4),
        p_valor=round(p_valor, 6),
        alpha=alpha,
        rejeita_h0=rejeita,
        conclusao=conclusao,
    )


def teste_ks(
    df: pd.DataFrame, coluna: str, alpha: float = ALPHA_PADRAO
) -> ResultadoTeste:
    """Teste Kolmogorov-Smirnov para comparar distribuições entre os dois grupos de TARGET.

    H0: As distribuições de ``coluna`` em TARGET=0 e TARGET=1 são idênticas.
    H1: As distribuições diferem.

    Sensível a diferenças em qualquer ponto da distribuição (não só na média),
    útil para variáveis como EXT_SOURCE_1/2/3.

    Args:
        df: DataFrame com as colunas ``coluna`` e ``TARGET``.
        coluna: Nome da variável contínua.
        alpha: Nível de significância (padrão: 0.05).

    Returns:
        ResultadoTeste com estatística KS, p-valor e conclusão.

    Examples:
        >>> resultado = teste_ks(df_train, "EXT_SOURCE_2")
        >>> print(resultado)
    """
    validos = df[[coluna, TARGET_COL]].dropna()
    grupo_0 = validos.loc[validos[TARGET_COL] == 0, coluna].values
    grupo_1 = validos.loc[validos[TARGET_COL] == 1, coluna].values

    stat, p_valor = stats.ks_2samp(grupo_0, grupo_1)
    rejeita = p_valor < alpha

    if rejeita:
        conclusao = (
            f"'{coluna}' tem distribuições estatisticamente diferentes entre grupos "
            f"(KS={stat:.4f}; p={p_valor:.2e} < {alpha}). "
            f"Boa candidata como feature preditiva."
        )
    else:
        conclusao = (
            f"'{coluna}' não apresenta diferença distribucional significativa "
            f"entre grupos (p={p_valor:.2e} >= {alpha})."
        )

    return ResultadoTeste(
        nome_teste="Kolmogorov-Smirnov",
        variavel=coluna,
        estatistica=round(float(stat), 4),
        p_valor=round(p_valor, 6),
        alpha=alpha,
        rejeita_h0=rejeita,
        conclusao=conclusao,
    )


def sumarizar_testes(
    df: pd.DataFrame,
    colunas_continuas: list[str] | None = None,
    colunas_categoricas: list[str] | None = None,
    alpha: float = ALPHA_PADRAO,
) -> pd.DataFrame:
    """Executa múltiplos testes de hipóteses e retorna um DataFrame com os resultados.

    Aplica Mann-Whitney U para colunas contínuas e Qui-Quadrado para categóricas.

    Args:
        df: DataFrame com coluna TARGET.
        colunas_continuas: Lista de colunas numéricas a testar. Se None, usa
            todas as numéricas com mais de 2 valores únicos (exceto TARGET).
        colunas_categoricas: Lista de colunas categóricas a testar. Se None,
            usa todas as colunas do tipo object/string.
        alpha: Nível de significância.

    Returns:
        DataFrame com colunas: variavel, teste, estatistica, p_valor,
        rejeita_h0 — ordenado por p_valor crescente.

    Examples:
        >>> sumario = sumarizar_testes(df_train)
        >>> print(sumario[sumario["rejeita_h0"]].head(20))
    """
    resultados = []

    if colunas_continuas is None:
        numericas = df.select_dtypes("number").columns.tolist()
        colunas_continuas = [
            c for c in numericas if c != TARGET_COL and df[c].nunique() > 2
        ]

    if colunas_categoricas is None:
        colunas_categoricas = df.select_dtypes("object").columns.tolist()

    for col in colunas_continuas:
        try:
            r = teste_mann_whitney(df, col, alpha=alpha)
            resultados.append({
                "variavel": r.variavel,
                "teste": r.nome_teste,
                "estatistica": r.estatistica,
                "p_valor": r.p_valor,
                "rejeita_h0": r.rejeita_h0,
            })
        except Exception:
            pass

    for col in colunas_categoricas:
        try:
            r = teste_chi2(df, col, alpha=alpha)
            resultados.append({
                "variavel": r.variavel,
                "teste": r.nome_teste,
                "estatistica": r.estatistica,
                "p_valor": r.p_valor,
                "rejeita_h0": r.rejeita_h0,
            })
        except Exception:
            pass

    return (
        pd.DataFrame(resultados)
        .sort_values("p_valor")
        .reset_index(drop=True)
    )
