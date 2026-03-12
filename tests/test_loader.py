"""
Testes unitários para src/data/loader.py.

Usa arquivos CSV temporários (tmp_path do pytest) — sem dependência
dos dados reais em data/. Garante que o loader funciona em qualquer
ambiente, inclusive CI/CD.
"""

import pandas as pd
import pytest

from src.data.loader import DataLoadError, HomeCreditDataLoader


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_valido(tmp_path) -> str:
    """Cria um CSV mínimo válido no diretório temporário do pytest."""
    df = pd.DataFrame({
        "SK_ID_CURR": [100001, 100002, 100003],
        "TARGET": [0, 1, 0],
        "AMT_CREDIT": [200000, 300000, 150000],
        "AMT_INCOME_TOTAL": [80000, 60000, 100000],
    })
    caminho = tmp_path / "application_train.csv"
    df.to_csv(caminho, index=False)
    return str(tmp_path)


@pytest.fixture
def loader_valido(csv_valido) -> HomeCreditDataLoader:
    return HomeCreditDataLoader(data_dir=csv_valido)


# ---------------------------------------------------------------------------
# Testes de inicialização
# ---------------------------------------------------------------------------

class TestHomeCreditDataLoaderInit:

    def test_inicializa_com_diretorio_valido(self, csv_valido):
        loader = HomeCreditDataLoader(data_dir=csv_valido)
        assert loader is not None

    def test_inicializa_com_diretorio_inexistente_sem_crash(self):
        # Não deve lançar exceção — apenas loga um warning
        loader = HomeCreditDataLoader(data_dir="/caminho/que/nao/existe")
        assert loader is not None


# ---------------------------------------------------------------------------
# Testes de carregamento bem-sucedido
# ---------------------------------------------------------------------------

class TestCarregamento:

    def test_load_application_train_retorna_dataframe(self, loader_valido):
        df = loader_valido.load_application_train()
        assert isinstance(df, pd.DataFrame)

    def test_load_application_train_tem_linhas(self, loader_valido):
        df = loader_valido.load_application_train()
        assert len(df) > 0

    def test_load_application_train_tem_colunas_esperadas(self, loader_valido):
        df = loader_valido.load_application_train()
        assert "SK_ID_CURR" in df.columns
        assert "TARGET" in df.columns

    def test_load_application_train_numero_correto_de_linhas(self, loader_valido):
        df = loader_valido.load_application_train()
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Testes de erro
# ---------------------------------------------------------------------------

class TestErros:

    def test_levanta_data_load_error_se_arquivo_nao_existe(self, tmp_path):
        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        with pytest.raises(DataLoadError):
            loader.load_application_train()

    def test_mensagem_de_erro_menciona_arquivo(self, tmp_path):
        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        with pytest.raises(DataLoadError, match="application_train.csv"):
            loader.load_application_train()

    def test_levanta_data_load_error_para_bureau(self, tmp_path):
        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        with pytest.raises(DataLoadError):
            loader.load_bureau()

    def test_levanta_data_load_error_para_previous_applications(self, tmp_path):
        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        with pytest.raises(DataLoadError):
            loader.load_previous_applications()

    def test_data_load_error_e_subclasse_de_exception(self):
        assert issubclass(DataLoadError, Exception)


# ---------------------------------------------------------------------------
# Testes de load_all_core_tables
# ---------------------------------------------------------------------------

class TestLoadAllCoreTables:

    def test_load_all_core_tables_levanta_erro_sem_arquivos(self, tmp_path):
        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        with pytest.raises(DataLoadError):
            loader.load_all_core_tables()

    def test_load_all_core_tables_retorna_dict_com_chaves_corretas(self, tmp_path):
        # Cria os três arquivos necessários
        for nome in ["application_train.csv", "bureau.csv", "previous_application.csv"]:
            pd.DataFrame({"SK_ID_CURR": [1, 2]}).to_csv(tmp_path / nome, index=False)

        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        resultado = loader.load_all_core_tables()

        assert isinstance(resultado, dict)
        assert "application_train" in resultado
        assert "bureau" in resultado
        assert "previous_application" in resultado

    def test_load_all_core_tables_valores_sao_dataframes(self, tmp_path):
        for nome in ["application_train.csv", "bureau.csv", "previous_application.csv"]:
            pd.DataFrame({"SK_ID_CURR": [1, 2]}).to_csv(tmp_path / nome, index=False)

        loader = HomeCreditDataLoader(data_dir=str(tmp_path))
        resultado = loader.load_all_core_tables()

        for chave, df in resultado.items():
            assert isinstance(df, pd.DataFrame), f"'{chave}' deveria ser um DataFrame"
