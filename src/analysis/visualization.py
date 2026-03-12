"""
Visualizações padronizadas para o dataset Home Credit Default Risk.

Todas as funções retornam o objeto Figure do Matplotlib, permitindo
salvar ou customizar externamente. Nenhuma função chama plt.show()
diretamente — isso é responsabilidade do chamador.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

TARGET_COL = "TARGET"
COR_PAGADOR = "#4CAF50"
COR_INADIMPLENTE = "#F44336"
PALETTE_TARGET = {0: COR_PAGADOR, 1: COR_INADIMPLENTE}


def plot_distribuicao_target(df: pd.DataFrame) -> plt.Figure:
    """Gráfico de barras com a distribuição absoluta e percentual do TARGET.

    Evidencia o desbalanceamento de classes — ponto de partida obrigatório
    para qualquer análise de crédito.

    Args:
        df: DataFrame contendo a coluna ``TARGET``.

    Returns:
        Figure do Matplotlib.

    Examples:
        >>> fig = plot_distribuicao_target(df_train)
        >>> fig.savefig("target_dist.png", dpi=150)
    """
    contagem = df[TARGET_COL].value_counts().sort_index()
    pct = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Adimplente (0)", "Inadimplente (1)"],
        contagem.values,
        color=[COR_PAGADOR, COR_INADIMPLENTE],
        edgecolor="white",
        width=0.5,
    )

    for bar, p in zip(bars, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + contagem.max() * 0.01,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Distribuição do TARGET\n(Adimplentes vs Inadimplentes)", fontsize=13)
    ax.set_ylabel("Número de Clientes")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


def plot_distribuicao_feature(
    df: pd.DataFrame,
    coluna: str,
    clip_percentil: float = 99.0,
) -> plt.Figure:
    """KDE e boxplot de uma feature contínua, separados por TARGET.

    Permite visualizar se a variável discrimina adimplentes de inadimplentes.

    Args:
        df: DataFrame com as colunas ``coluna`` e ``TARGET``.
        coluna: Nome da variável numérica.
        clip_percentil: Remove valores acima deste percentil para melhor
            visualização (padrão: 99 — remove 1% dos outliers extremos).

    Returns:
        Figure do Matplotlib com dois subplots (KDE + boxplot).

    Examples:
        >>> fig = plot_distribuicao_feature(df_train, "AMT_INCOME_TOTAL")
        >>> plt.show()
    """
    limite = df[coluna].quantile(clip_percentil / 100)
    dados = df[df[coluna] <= limite][[coluna, TARGET_COL]].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

    # KDE por TARGET
    sns.kdeplot(
        data=dados, x=coluna, hue=TARGET_COL,
        common_norm=False, fill=True, alpha=0.4,
        palette=PALETTE_TARGET, ax=axes[0],
    )
    mediana_0 = dados.loc[dados[TARGET_COL] == 0, coluna].median()
    mediana_1 = dados.loc[dados[TARGET_COL] == 1, coluna].median()
    axes[0].axvline(mediana_0, color=COR_PAGADOR, linestyle="--", linewidth=1.5,
                    label=f"Mediana adimplente = {mediana_0:,.2f}")
    axes[0].axvline(mediana_1, color=COR_INADIMPLENTE, linestyle="--", linewidth=1.5,
                    label=f"Mediana inadimplente = {mediana_1:,.2f}")
    axes[0].set_title(f"Distribuição de {coluna} por TARGET", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("")

    # Boxplot
    dados_plot = dados.copy()
    dados_plot["target_label"] = dados_plot[TARGET_COL].map({0: "Adimplente", 1: "Inadimplente"})
    sns.boxplot(
        data=dados_plot, x=coluna, y="target_label",
        hue="target_label",
        palette={"Adimplente": COR_PAGADOR, "Inadimplente": COR_INADIMPLENTE},
        orient="h", ax=axes[1], legend=False,
    )
    axes[1].set_ylabel("")
    axes[1].set_xlabel(coluna)

    fig.tight_layout()
    return fig


def plot_correlacao_heatmap(
    df: pd.DataFrame,
    colunas: list[str] | None = None,
) -> plt.Figure:
    """Heatmap de correlação de Pearson entre variáveis numéricas.

    Args:
        df: DataFrame com os dados.
        colunas: Lista de colunas a incluir. Se None, usa as colunas
            numéricas com menos de 50% de nulos.

    Returns:
        Figure do Matplotlib.

    Examples:
        >>> colunas = ["TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "EXT_SOURCE_2"]
        >>> fig = plot_correlacao_heatmap(df_train, colunas)
    """
    if colunas is None:
        numericas = df.select_dtypes("number")
        colunas = numericas.columns[numericas.isnull().mean() < 0.5].tolist()

    matriz = df[colunas].corr()
    mask = np.triu(np.ones_like(matriz, dtype=bool))

    n = len(colunas)
    tamanho = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(tamanho, tamanho * 0.85))

    sns.heatmap(
        matriz, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.4,
        annot_kws={"size": max(6, 10 - n // 5)},
        ax=ax,
    )
    ax.set_title("Heatmap de Correlação (Pearson)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_taxa_por_categoria(
    df: pd.DataFrame,
    coluna: str,
    min_amostras: int = 30,
) -> plt.Figure:
    """Barplot horizontal com taxa de inadimplência por categoria.

    Inclui linha vertical com a média geral e rótulos com % e n.

    Args:
        df: DataFrame com as colunas ``coluna`` e ``TARGET``.
        coluna: Nome da variável categórica.
        min_amostras: Categorias com menos amostras são excluídas.

    Returns:
        Figure do Matplotlib.

    Examples:
        >>> fig = plot_taxa_por_categoria(df_train, "NAME_INCOME_TYPE")
        >>> plt.show()
    """
    taxa = (
        df.groupby(coluna, observed=True)[TARGET_COL]
        .agg(["mean", "count"])
        .reset_index()
    )
    taxa = taxa[taxa["count"] >= min_amostras].copy()
    taxa["mean"] *= 100
    taxa = taxa.sort_values("mean", ascending=True)

    media_geral = df[TARGET_COL].mean() * 100
    altura = max(4, len(taxa) * 0.5)
    fig, ax = plt.subplots(figsize=(10, altura))

    bars = ax.barh(taxa[coluna], taxa["mean"], color="#E57373", edgecolor="white")
    ax.axvline(
        media_geral, color="black", linestyle="--", linewidth=1.2,
        label=f"Média geral: {media_geral:.1f}%",
    )

    for bar, (_, row) in zip(bars, taxa.iterrows()):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean']:.1f}%  (n={int(row['count']):,})",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Taxa de Inadimplência (%)")
    ax.set_title(f"Taxa de Inadimplência por {coluna}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(right=taxa["mean"].max() * 1.4)
    fig.tight_layout()
    return fig


def plot_ext_sources(df: pd.DataFrame) -> plt.Figure:
    """KDE das três fontes externas (EXT_SOURCE_1/2/3) separadas por TARGET.

    Essas variáveis costumam ser as mais preditivas do dataset e merecem
    um gráfico dedicado.

    Args:
        df: DataFrame com colunas EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 e TARGET.

    Returns:
        Figure do Matplotlib com três subplots lado a lado.

    Examples:
        >>> fig = plot_ext_sources(df_train)
        >>> plt.show()
    """
    ext_sources = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]

    fig, axes = plt.subplots(1, len(ext_sources), figsize=(6 * len(ext_sources), 5))
    if len(ext_sources) == 1:
        axes = [axes]

    for ax, col in zip(axes, ext_sources):
        dados = df[[col, TARGET_COL]].dropna()
        sns.kdeplot(
            data=dados, x=col, hue=TARGET_COL,
            common_norm=False, fill=True, alpha=0.4,
            palette=PALETTE_TARGET, ax=ax,
        )
        m0 = dados.loc[dados[TARGET_COL] == 0, col].median()
        m1 = dados.loc[dados[TARGET_COL] == 1, col].median()
        ax.axvline(m0, color=COR_PAGADOR, linestyle="--", linewidth=1.5)
        ax.axvline(m1, color=COR_INADIMPLENTE, linestyle="--", linewidth=1.5)
        ax.set_title(f"{col}\n(med. adim.={m0:.2f} | inadim.={m1:.2f})", fontsize=10)

    fig.suptitle("Scores Externos por TARGET (verde=adimplente, vermelho=inadimplente)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_perfil_nulos(df: pd.DataFrame, top_n: int = 40) -> plt.Figure:
    """Barplot horizontal com as colunas com maior percentual de valores nulos.

    Args:
        df: DataFrame a analisar.
        top_n: Número máximo de colunas a exibir (padrão: 40).

    Returns:
        Figure do Matplotlib.

    Examples:
        >>> fig = plot_perfil_nulos(df_train)
    """
    nulos_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    nulos_pct = nulos_pct[nulos_pct > 0].head(top_n)

    altura = max(6, len(nulos_pct) * 0.3)
    fig, ax = plt.subplots(figsize=(10, altura))

    ax.barh(nulos_pct.index[::-1], nulos_pct.values[::-1], color="steelblue")
    ax.axvline(50, color="red", linestyle="--", linewidth=1.2, label="Limite 50%")
    ax.set_xlabel("% de Valores Nulos")
    ax.set_title(f"Top {len(nulos_pct)} Colunas com Mais Valores Nulos", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
