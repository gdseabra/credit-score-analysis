"""
Estatística descritiva para o dataset Home Credit Default Risk.

Funções reutilizáveis que encapsulam as análises feitas no notebook 01,
permitindo reprodução programática sem dependência do Jupyter.
"""

import numpy as np
import pandas as pd

TARGET_COL = "TARGET"


def resumo_estatistico(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatísticas descritivas completas para todas as colunas numéricas.

    Inclui média, mediana, desvio-padrão, skewness, kurtosis e percentual de nulos.

    Args:
        df: DataFrame com os dados brutos do Home Credit.

    Returns:
        DataFrame com índice = nome da coluna e colunas de estatísticas.

    Examples:
        >>> resumo = resumo_estatistico(df_train)
        >>> resumo.sort_values("skewness", ascending=False).head(10)
    """
    numericas = df.select_dtypes("number").columns.tolist()
    if TARGET_COL in numericas:
        numericas.remove(TARGET_COL)

    stats = df[numericas].agg(["mean", "median", "std", "min", "max"]).T
    stats.columns = ["media", "mediana", "desvio_padrao", "minimo", "maximo"]
    stats["skewness"] = df[numericas].skew()
    stats["kurtosis"] = df[numericas].kurt()
    stats["pct_nulos"] = (df[numericas].isnull().mean() * 100).round(2)
    stats["n_unicos"] = df[numericas].nunique()

    return stats.round(4)


def taxa_inadimplencia_por_grupo(
    df: pd.DataFrame, coluna: str, min_amostras: int = 30
) -> pd.DataFrame:
    """Calcula a taxa de inadimplência (TARGET=1) para cada categoria de uma coluna.

    Útil para interpretar métricas aplicadas a negócio: identifica segmentos
    de clientes com maior risco de crédito.

    Args:
        df: DataFrame contendo as colunas ``coluna`` e ``TARGET``.
        coluna: Nome da coluna categórica ou binária para agrupar.
        min_amostras: Grupos com menos amostras que este valor são filtrados.

    Returns:
        DataFrame com colunas: categoria, n_clientes, n_inadimplentes,
        taxa_inadimplencia_pct, diferenca_media_pct.

    Raises:
        KeyError: Se ``coluna`` ou ``TARGET`` não existirem no DataFrame.

    Examples:
        >>> taxa_inadimplencia_por_grupo(df_train, "NAME_INCOME_TYPE")
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Coluna '{TARGET_COL}' não encontrada no DataFrame.")
    if coluna not in df.columns:
        raise KeyError(f"Coluna '{coluna}' não encontrada no DataFrame.")

    media_geral = df[TARGET_COL].mean() * 100

    resultado = (
        df.groupby(coluna, observed=True)[TARGET_COL]
        .agg(n_clientes="count", n_inadimplentes="sum")
        .reset_index()
    )
    resultado["taxa_inadimplencia_pct"] = (
        resultado["n_inadimplentes"] / resultado["n_clientes"] * 100
    ).round(2)
    resultado["diferenca_media_pct"] = (
        resultado["taxa_inadimplencia_pct"] - media_geral
    ).round(2)

    resultado = resultado[resultado["n_clientes"] >= min_amostras]
    resultado = resultado.sort_values("taxa_inadimplencia_pct", ascending=False)
    resultado = resultado.rename(columns={coluna: "categoria"})

    return resultado.reset_index(drop=True)


def correlacao_com_target(
    df: pd.DataFrame, metodo: str = "pearson", top_n: int | None = None
) -> pd.Series:
    """Calcula a correlação de todas as colunas numéricas com TARGET.

    Args:
        df: DataFrame contendo a coluna ``TARGET``.
        metodo: Método de correlação: ``'pearson'``, ``'spearman'`` ou ``'kendall'``.
        top_n: Se fornecido, retorna apenas as ``top_n`` correlações de maior
               magnitude (positivas e negativas combinadas).

    Returns:
        Series com índice = nome da coluna e valores = correlação com TARGET,
        ordenada do menor (proteção) ao maior (risco).

    Examples:
        >>> corr = correlacao_com_target(df_train, top_n=20)
        >>> print(corr)
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Coluna '{TARGET_COL}' não encontrada no DataFrame.")

    numericas = df.select_dtypes("number")
    corr = (
        numericas.corr(method=metodo)[TARGET_COL]
        .drop(TARGET_COL)
        .dropna()
        .sort_values()
    )

    if top_n is not None:
        corr = pd.concat([corr.head(top_n), corr.tail(top_n)])

    return corr


def perfil_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna um perfil completo dos valores nulos do DataFrame.

    Args:
        df: DataFrame a ser analisado.

    Returns:
        DataFrame com colunas: coluna, n_nulos, pct_nulos, dtype — ordenado
        por percentual de nulos decrescente. Inclui apenas colunas com pelo
        menos 1 nulo.

    Examples:
        >>> perfil = perfil_nulos(df_train)
        >>> cols_altas = perfil[perfil["pct_nulos"] > 50]
    """
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]

    resultado = pd.DataFrame({
        "coluna": nulos.index,
        "n_nulos": nulos.values,
        "pct_nulos": (df[nulos.index].isnull().mean() * 100).round(2).values,
        "dtype": df[nulos.index].dtypes.astype(str).values,
    })

    resultado = resultado.sort_values("pct_nulos", ascending=False).reset_index(drop=True)

    total = len(df.columns)
    com_nulo = len(resultado)
    print(
        f"Colunas com nulos: {com_nulo} de {total} "
        f"({com_nulo / total * 100:.1f}%)\n"
        f"Colunas com >50% nulos: {(resultado['pct_nulos'] > 50).sum()}"
    )

    return resultado
