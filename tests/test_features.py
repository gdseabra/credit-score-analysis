"""
Testes unitários para src/features/build_features.py.

Cobre: AnomalyHandler, DomainFeatureBuilder e build_preprocessor_pipeline.
Todos os testes usam dados sintéticos do conftest.py — sem dependência de arquivos CSV.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.build_features import (
    ANOMALY_DAYS_EMPLOYED,
    DAYS_IN_YEAR,
    AnomalyHandler,
    DomainFeatureBuilder,
    build_preprocessor_pipeline,
)


# ---------------------------------------------------------------------------
# AnomalyHandler
# ---------------------------------------------------------------------------

class TestAnomalyHandler:

    def test_cria_coluna_flag(self, df_com_anomalia_isolada):
        result = AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert "DAYS_EMPLOYED_ANOMALY" in result.columns

    def test_flag_vale_1_para_anomalia(self, df_com_anomalia_isolada):
        result = AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert result.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 1

    def test_flag_vale_0_para_valores_normais(self, df_com_anomalia_isolada):
        result = AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert result.loc[1, "DAYS_EMPLOYED_ANOMALY"] == 0
        assert result.loc[2, "DAYS_EMPLOYED_ANOMALY"] == 0

    def test_substitui_anomalia_por_nan(self, df_com_anomalia_isolada):
        result = AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert pd.isna(result.loc[0, "DAYS_EMPLOYED"])

    def test_preserva_valores_normais(self, df_com_anomalia_isolada):
        result = AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert result.loc[1, "DAYS_EMPLOYED"] == -500
        assert result.loc[2, "DAYS_EMPLOYED"] == -1000

    def test_nao_modifica_dataframe_original(self, df_com_anomalia_isolada):
        original = df_com_anomalia_isolada["DAYS_EMPLOYED"].iloc[0]
        AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert df_com_anomalia_isolada["DAYS_EMPLOYED"].iloc[0] == original

    def test_quantidade_de_anomalias_corretas(self, df_minimo):
        result = AnomalyHandler().fit_transform(df_minimo)
        assert result["DAYS_EMPLOYED_ANOMALY"].sum() == 2

    def test_levanta_keyerror_sem_coluna(self):
        df_sem_coluna = pd.DataFrame({"OUTRA_COLUNA": [1, 2, 3]})
        with pytest.raises(KeyError, match="DAYS_EMPLOYED"):
            AnomalyHandler().fit_transform(df_sem_coluna)

    def test_fit_retorna_self(self, df_com_anomalia_isolada):
        handler = AnomalyHandler()
        resultado = handler.fit(df_com_anomalia_isolada)
        assert resultado is handler

    def test_tipo_da_flag_e_int8(self, df_com_anomalia_isolada):
        result = AnomalyHandler().fit_transform(df_com_anomalia_isolada)
        assert result["DAYS_EMPLOYED_ANOMALY"].dtype == np.int8


# ---------------------------------------------------------------------------
# DomainFeatureBuilder
# ---------------------------------------------------------------------------

class TestDomainFeatureBuilder:

    def test_cria_credit_income_ratio(self, df_minimo):
        result = DomainFeatureBuilder().fit_transform(df_minimo)
        assert "CREDIT_INCOME_RATIO" in result.columns

    def test_cria_todas_as_features(self, df_minimo):
        features_esperadas = [
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "CREDIT_TERM_MONTHS",
            "INCOME_PER_FAMILY_MEMBER",
            "AGE_YEARS",
            "EMPLOYED_YEARS",
            "EMPLOYED_TO_AGE_RATIO",
        ]
        result = DomainFeatureBuilder().fit_transform(df_minimo)
        for feature in features_esperadas:
            assert feature in result.columns, f"Feature ausente: {feature}"

    def test_credit_income_ratio_calculado_corretamente(self):
        df = pd.DataFrame({
            "DAYS_EMPLOYED": [-500],
            "DAYS_BIRTH": [-10000],
            "AMT_CREDIT": [300000],
            "AMT_INCOME_TOTAL": [100000],
            "AMT_ANNUITY": [15000],
            "CNT_FAM_MEMBERS": [3],
        })
        result = DomainFeatureBuilder().fit_transform(df)
        assert result.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(3.0)

    def test_age_years_calculado_corretamente(self):
        df = pd.DataFrame({
            "DAYS_EMPLOYED": [-500],
            "DAYS_BIRTH": [-int(30 * DAYS_IN_YEAR)],
            "AMT_CREDIT": [200000],
            "AMT_INCOME_TOTAL": [80000],
            "AMT_ANNUITY": [10000],
            "CNT_FAM_MEMBERS": [2],
        })
        result = DomainFeatureBuilder().fit_transform(df)
        assert result.loc[0, "AGE_YEARS"] == pytest.approx(30.0, abs=0.1)

    def test_sem_divisao_por_zero_annuity(self):
        df = pd.DataFrame({
            "DAYS_EMPLOYED": [-500],
            "DAYS_BIRTH": [-10000],
            "AMT_CREDIT": [200000],
            "AMT_INCOME_TOTAL": [80000],
            "AMT_ANNUITY": [0],  # divisão por zero!
            "CNT_FAM_MEMBERS": [2],
        })
        result = DomainFeatureBuilder().fit_transform(df)
        assert pd.isna(result.loc[0, "CREDIT_TERM_MONTHS"])

    def test_sem_divisao_por_zero_familia(self):
        df = pd.DataFrame({
            "DAYS_EMPLOYED": [-500],
            "DAYS_BIRTH": [-10000],
            "AMT_CREDIT": [200000],
            "AMT_INCOME_TOTAL": [80000],
            "AMT_ANNUITY": [10000],
            "CNT_FAM_MEMBERS": [0],  # divisão por zero!
        })
        result = DomainFeatureBuilder().fit_transform(df)
        assert pd.isna(result.loc[0, "INCOME_PER_FAMILY_MEMBER"])

    def test_nao_modifica_dataframe_original(self, df_minimo):
        colunas_originais = set(df_minimo.columns)
        DomainFeatureBuilder().fit_transform(df_minimo)
        assert set(df_minimo.columns) == colunas_originais

    def test_fit_retorna_self(self, df_minimo):
        builder = DomainFeatureBuilder()
        resultado = builder.fit(df_minimo)
        assert resultado is builder

    def test_feature_descriptions_retorna_dataframe(self):
        desc = DomainFeatureBuilder.feature_descriptions()
        assert isinstance(desc, pd.DataFrame)
        assert "feature" in desc.columns
        assert "descricao" in desc.columns
        assert len(desc) == 7


# ---------------------------------------------------------------------------
# build_preprocessor_pipeline
# ---------------------------------------------------------------------------

class TestBuildPreprocessorPipeline:

    def test_retorna_pipeline_sklearn(self):
        pipeline = build_preprocessor_pipeline(
            numeric_cols=["AMT_CREDIT"],
            categorical_cols=["CODE_GENDER"],
        )
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_tem_tres_etapas(self):
        pipeline = build_preprocessor_pipeline(
            numeric_cols=["AMT_CREDIT"],
            categorical_cols=["CODE_GENDER"],
        )
        assert len(pipeline.steps) == 3

    def test_pipeline_fit_transform_executa_sem_erro(self, df_com_target):
        numeric_cols = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                        "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS"]
        categorical_cols = ["CODE_GENDER"]

        X = df_com_target.drop(columns=["TARGET"])
        pipeline = build_preprocessor_pipeline(numeric_cols, categorical_cols)
        resultado = pipeline.fit_transform(X)

        assert resultado is not None
        assert resultado.shape[0] == len(df_com_target)

    def test_pipeline_sem_nulos_apos_transform(self, df_com_target):
        numeric_cols = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                        "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS"]
        categorical_cols = ["CODE_GENDER"]

        X = df_com_target.drop(columns=["TARGET"])
        pipeline = build_preprocessor_pipeline(numeric_cols, categorical_cols)
        resultado = pipeline.fit_transform(X)

        import numpy as np
        assert not np.isnan(resultado).any(), "Pipeline não deve deixar valores nulos"
