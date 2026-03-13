"""
Feature Engineering pipeline para o dataset Home Credit Default Risk.

Módulo estruturado seguindo os princípios SOLID e o padrão Scikit-Learn,
garantindo compatibilidade com sklearn.pipeline.Pipeline e ausência de
data leakage entre treino e teste.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes de domínio — sem magic numbers no código
# ---------------------------------------------------------------------------

ANOMALY_DAYS_EMPLOYED: int = 365243
"""Valor sentinela usado pelo sistema legado para clientes sem emprego formal."""

DAYS_IN_YEAR: float = 365.25
"""Divisor padrão para converter dias em anos, considerando anos bissextos."""


# ---------------------------------------------------------------------------
# Passo 2 — Tratamento de Anomalias
# ---------------------------------------------------------------------------

class AnomalyHandler(BaseEstimator, TransformerMixin):
    """Detecta e trata anomalias conhecidas no dataset Home Credit.

    Atualmente trata a anomalia em DAYS_EMPLOYED, onde o valor sentinel
    ANOMALY_DAYS_EMPLOYED (365243) representa clientes sem vínculo empregatício
    formal e foi inserido pelo sistema legado no lugar de NaN.

    Compatível com sklearn.pipeline.Pipeline.

    Attributes:
        anomaly_days_employed (int): Valor sentinel a ser substituído por NaN.

    Examples:
        >>> handler = AnomalyHandler()
        >>> df_clean = handler.fit_transform(df_raw)
    """

    def __init__(self, anomaly_days_employed: int = ANOMALY_DAYS_EMPLOYED) -> None:
        self.anomaly_days_employed = anomaly_days_employed

    def fit(self, X: pd.DataFrame, y=None) -> "AnomalyHandler":
        """Sem estado para aprender — retorna self para compatibilidade com Pipeline.

        Args:
            X: DataFrame de entrada.
            y: Ignorado. Presente para compatibilidade com a API do Scikit-Learn.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica o tratamento de anomalias no DataFrame.

        Operações realizadas:
        - Cria a flag binária ``DAYS_EMPLOYED_ANOMALY`` (1 se for anomalia, 0 caso contrário).
        - Substitui o valor sentinel em ``DAYS_EMPLOYED`` por ``np.nan``.

        Args:
            X: DataFrame contendo a coluna ``DAYS_EMPLOYED``.
            y: Ignorado. Presente para compatibilidade com a API do Scikit-Learn.

        Returns:
            Cópia do DataFrame com a flag criada e a anomalia substituída por NaN.

        Raises:
            KeyError: Se a coluna ``DAYS_EMPLOYED`` não existir no DataFrame.
        """
        if "DAYS_EMPLOYED" not in X.columns:
            raise KeyError("Coluna 'DAYS_EMPLOYED' não encontrada no DataFrame.")

        X_out = X.copy()
        X_out["DAYS_EMPLOYED_ANOMALY"] = (
            X_out["DAYS_EMPLOYED"] == self.anomaly_days_employed
        ).astype(np.int8)
        X_out.loc[X_out["DAYS_EMPLOYED"] == self.anomaly_days_employed, "DAYS_EMPLOYED"] = np.nan

        return X_out


# ---------------------------------------------------------------------------
# Passo 3 — Features de Domínio (Negócio)
# ---------------------------------------------------------------------------

class DomainFeatureBuilder(BaseEstimator, TransformerMixin):
    """Cria features derivadas com base em conhecimento de domínio de crédito.

    Todas as features criadas foram validadas na EDA (notebook 01) e têm
    justificativa de negócio explícita nos atributos ``_FEATURE_DESCRIPTIONS``.

    Compatível com sklearn.pipeline.Pipeline.

    Examples:
        >>> builder = DomainFeatureBuilder()
        >>> df_enriched = builder.fit_transform(df_clean)
    """

    _FEATURE_DESCRIPTIONS: dict[str, str] = {
        "CREDIT_INCOME_RATIO": "Crédito total / Renda anual. Razão alta indica superendividamento.",
        "ANNUITY_INCOME_RATIO": "Parcela mensal / Renda anual. Mede o comprometimento de renda.",
        "CREDIT_TERM_MONTHS": "Crédito / Parcela. Prazo implícito do empréstimo em meses.",
        "INCOME_PER_FAMILY_MEMBER": "Renda / Membros da família. Proxy de capacidade de pagamento real.",
        "AGE_YEARS": "Idade do cliente convertida de dias para anos.",
        "EMPLOYED_YEARS": "Tempo de emprego convertido de dias para anos (após limpeza de anomalia).",
        "EMPLOYED_TO_AGE_RATIO": "Tempo empregado / Idade. Estabilidade relativa no emprego.",
    }

    def fit(self, X: pd.DataFrame, y=None) -> "DomainFeatureBuilder":
        """Sem estado para aprender — retorna self para compatibilidade com Pipeline.

        Args:
            X: DataFrame de entrada.
            y: Ignorado. Presente para compatibilidade com a API do Scikit-Learn.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Cria todas as features de domínio no DataFrame.

        Args:
            X: DataFrame contendo as colunas brutas do Home Credit.
            y: Ignorado. Presente para compatibilidade com a API do Scikit-Learn.

        Returns:
            Cópia do DataFrame original enriquecida com as novas features.
            O DataFrame original **não é modificado** (operação sem side-effects).
        """
        X_out = X.copy()

        # --- Ratios financeiros ---
        X_out["CREDIT_INCOME_RATIO"] = (
            X_out["AMT_CREDIT"] / X_out["AMT_INCOME_TOTAL"]
        )
        X_out["ANNUITY_INCOME_RATIO"] = (
            X_out["AMT_ANNUITY"] / X_out["AMT_INCOME_TOTAL"]
        )
        # Evita divisão por zero caso AMT_ANNUITY seja nulo ou zero
        X_out["CREDIT_TERM_MONTHS"] = X_out["AMT_CREDIT"] / X_out["AMT_ANNUITY"].replace(0, np.nan)

        # --- Features demográficas / emprego ---
        X_out["AGE_YEARS"] = X_out["DAYS_BIRTH"] / -DAYS_IN_YEAR
        X_out["EMPLOYED_YEARS"] = X_out["DAYS_EMPLOYED"] / -DAYS_IN_YEAR

        X_out["EMPLOYED_TO_AGE_RATIO"] = (
            X_out["EMPLOYED_YEARS"] / X_out["AGE_YEARS"]
        )

        # --- Renda per capita familiar ---
        X_out["INCOME_PER_FAMILY_MEMBER"] = (
            X_out["AMT_INCOME_TOTAL"] / X_out["CNT_FAM_MEMBERS"].replace(0, np.nan)
        )

        return X_out

    @classmethod
    def feature_descriptions(cls) -> pd.DataFrame:
        """Retorna um DataFrame com a documentação de cada feature criada.

        Returns:
            DataFrame com colunas ``feature`` e ``descricao``.
        """
        return pd.DataFrame(
            list(cls._FEATURE_DESCRIPTIONS.items()),
            columns=["feature", "descricao"],
        )


# ---------------------------------------------------------------------------
# Passo 4 — Pipeline de Pré-processamento Completo
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Passo 3b — Features de Tabelas Auxiliares (Bureau + Cartão + Parcelas)
# ---------------------------------------------------------------------------

class AuxiliaryFeatureBuilder(BaseEstimator, TransformerMixin):
    """Cria agregações das tabelas auxiliares do Home Credit e as junta ao dataset principal.

    Recebe os DataFrames auxiliares no construtor e, durante o ``transform``,
    calcula as agregações por ``SK_ID_CURR`` e realiza um LEFT JOIN no DataFrame
    principal. Nulos resultantes de clientes sem histórico são imputados downstream
    pelo pipeline de pré-processamento — esta classe não imputa.

    Compatível com sklearn.pipeline.Pipeline.

    Attributes:
        bureau (pd.DataFrame | None): Dados do bureau de crédito externo.
        credit_card (pd.DataFrame | None): Saldos mensais de cartão de crédito.
        installments (pd.DataFrame | None): Histórico de pagamentos de parcelas.

    Examples:
        >>> builder = AuxiliaryFeatureBuilder(bureau=df_bureau,
        ...                                   credit_card=df_cc,
        ...                                   installments=df_inst)
        >>> df_enriched = builder.fit_transform(df_application)
    """

    # Nomes das colunas de origem usadas nas agregações
    _COL_SK_ID_CURR: str = "SK_ID_CURR"

    # Bureau
    _COL_BUREAU_ID: str = "SK_ID_BUREAU"
    _COL_CREDIT_ACTIVE: str = "CREDIT_ACTIVE"
    _COL_CREDIT_ACTIVE_VALUE: str = "Active"
    _COL_AMT_CREDIT_SUM: str = "AMT_CREDIT_SUM"
    _COL_DAYS_CREDIT: str = "DAYS_CREDIT"

    # Credit card
    _COL_AMT_BALANCE: str = "AMT_BALANCE"
    _COL_AMT_CREDIT_LIMIT: str = "AMT_CREDIT_LIMIT_ACTUAL"
    _COL_AMT_DRAWINGS: str = "AMT_DRAWINGS_CURRENT"

    # Installments
    _COL_AMT_PAYMENT: str = "AMT_PAYMENT"
    _COL_AMT_INSTALMENT: str = "AMT_INSTALMENT"
    _COL_DAYS_ENTRY_PAYMENT: str = "DAYS_ENTRY_PAYMENT"
    _COL_DAYS_INSTALMENT: str = "DAYS_INSTALMENT"

    def __init__(
        self,
        bureau: pd.DataFrame | None = None,
        credit_card: pd.DataFrame | None = None,
        installments: pd.DataFrame | None = None,
    ) -> None:
        self.bureau = bureau
        self.credit_card = credit_card
        self.installments = installments

    def fit(self, X: pd.DataFrame, y=None) -> "AuxiliaryFeatureBuilder":
        """Sem estado para aprender — retorna self para compatibilidade com Pipeline.

        Args:
            X: DataFrame de entrada.
            y: Ignorado. Presente para compatibilidade com a API do Scikit-Learn.

        Returns:
            self
        """
        return self

    def _agregar_bureau(self) -> pd.DataFrame:
        """Calcula agregações da tabela bureau por cliente.

        Returns:
            DataFrame indexado por SK_ID_CURR com as colunas:
            bureau_loan_count, bureau_active_loans,
            bureau_total_credit_sum, bureau_mean_days_credit.
        """
        df = self.bureau.copy()
        agg = df.groupby(self._COL_SK_ID_CURR).agg(
            bureau_loan_count=(self._COL_BUREAU_ID, "count"),
            bureau_active_loans=(
                self._COL_CREDIT_ACTIVE,
                lambda s: (s == self._COL_CREDIT_ACTIVE_VALUE).sum(),
            ),
            bureau_total_credit_sum=(self._COL_AMT_CREDIT_SUM, "sum"),
            bureau_mean_days_credit=(self._COL_DAYS_CREDIT, "mean"),
        ).reset_index()
        return agg

    def _agregar_credit_card(self) -> pd.DataFrame:
        """Calcula agregações da tabela credit_card_balance por cliente.

        Returns:
            DataFrame indexado por SK_ID_CURR com as colunas:
            cc_utilization_mean, cc_drawings_total, cc_months_with_balance.
        """
        df = self.credit_card.copy()

        # Taxa de utilização — trata divisão por zero substituindo limite=0 por NaN
        df["_utilization"] = df[self._COL_AMT_BALANCE] / df[
            self._COL_AMT_CREDIT_LIMIT
        ].replace(0, np.nan)

        agg = df.groupby(self._COL_SK_ID_CURR).agg(
            cc_utilization_mean=("_utilization", "mean"),
            cc_drawings_total=(self._COL_AMT_DRAWINGS, "sum"),
            cc_months_with_balance=(
                self._COL_AMT_BALANCE,
                lambda s: (s > 0).sum(),
            ),
        ).reset_index()
        return agg

    def _agregar_installments(self) -> pd.DataFrame:
        """Calcula agregações da tabela installments_payments por cliente.

        Returns:
            DataFrame indexado por SK_ID_CURR com as colunas:
            install_payment_ratio_mean, install_days_late_mean, install_late_count.
        """
        df = self.installments.copy()

        # Razão pagamento/parcela — trata divisão por zero
        df["_payment_ratio"] = df[self._COL_AMT_PAYMENT] / df[
            self._COL_AMT_INSTALMENT
        ].replace(0, np.nan)

        # Dias de atraso (clamp em 0 para não contar pagamentos antecipados)
        df["_days_late"] = (
            df[self._COL_DAYS_ENTRY_PAYMENT] - df[self._COL_DAYS_INSTALMENT]
        ).clip(lower=0)

        agg = df.groupby(self._COL_SK_ID_CURR).agg(
            install_payment_ratio_mean=("_payment_ratio", "mean"),
            install_days_late_mean=("_days_late", "mean"),
            install_late_count=("_days_late", lambda s: (s > 0).sum()),
        ).reset_index()
        return agg

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Junta as agregações auxiliares ao DataFrame principal via LEFT JOIN.

        Clientes sem histórico nas tabelas auxiliares receberão NaN nas colunas
        correspondentes — a imputação é responsabilidade do pipeline downstream.

        Args:
            X: DataFrame principal contendo a coluna ``SK_ID_CURR``.
            y: Ignorado. Presente para compatibilidade com a API do Scikit-Learn.

        Returns:
            Cópia do DataFrame enriquecida com as features auxiliares.

        Raises:
            KeyError: Se a coluna ``SK_ID_CURR`` não existir no DataFrame.
        """
        if self._COL_SK_ID_CURR not in X.columns:
            raise KeyError(f"Coluna '{self._COL_SK_ID_CURR}' não encontrada no DataFrame.")

        X_out = X.copy()

        if self.bureau is not None:
            agg_bureau = self._agregar_bureau()
            X_out = X_out.merge(agg_bureau, on=self._COL_SK_ID_CURR, how="left")

        if self.credit_card is not None:
            agg_cc = self._agregar_credit_card()
            X_out = X_out.merge(agg_cc, on=self._COL_SK_ID_CURR, how="left")

        if self.installments is not None:
            agg_inst = self._agregar_installments()
            X_out = X_out.merge(agg_inst, on=self._COL_SK_ID_CURR, how="left")

        return X_out


def build_preprocessor_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Constrói o pipeline completo de pré-processamento compatível com Scikit-Learn.

    O pipeline aplica, na ordem:
    1. ``AnomalyHandler``    — trata a anomalia em DAYS_EMPLOYED.
    2. ``DomainFeatureBuilder`` — cria as features de negócio.
    3. ``ColumnTransformer``    — imputa nulos, escala numéricas e codifica categóricas.

    O ``ColumnTransformer`` usa ``remainder='drop'`` para garantir que apenas as
    colunas explicitamente declaradas passem para o modelo, evitando vazamento de
    features irrelevantes.

    Args:
        numeric_cols: Lista com os nomes das colunas numéricas a processar.
        categorical_cols: Lista com os nomes das colunas categóricas a processar.

    Returns:
        Pipeline do Scikit-Learn pronto para ser ajustado com ``.fit()``.

    Examples:
        >>> from src.features.build_features import build_preprocessor_pipeline
        >>> pipeline = build_preprocessor_pipeline(numeric_cols, categorical_cols)
        >>> pipeline.fit(df_train.drop("TARGET", axis=1), df_train["TARGET"])
        >>> X_transformed = pipeline.transform(df_test)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[
        ("anomaly_handler", AnomalyHandler()),
        ("domain_features", DomainFeatureBuilder()),
        ("preprocessor", column_transformer),
    ])

    return pipeline
