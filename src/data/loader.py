import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Configuração básica de logging (idealmente ficaria no src/utils/logger.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DataLoadError(Exception):
    """Exceção customizada para erros de carregamento de dados."""
    pass

class HomeCreditDataLoader:
    """
    Classe responsável por carregar os dados brutos da base do Home Credit.
    
    Princípio de Responsabilidade Única (SRP): Esta classe não limpa, 
    não treina modelos e não salva no banco. Ela APENAS carrega dados do disco.
    """

    def __init__(self, data_dir: str = "data/"):
        """
        Inicializa o carregador de dados.
        
        Args:
            data_dir (str): Caminho para o diretório onde os CSVs estão salvos.
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Valida se o diretório existe
        if not self.data_dir.exists():
            self.logger.warning(f"O diretório {self.data_dir} não foi encontrado.")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """
        Método privado e genérico para ler um arquivo CSV com tratamento de erros.
        
        Args:
            filename (str): Nome do arquivo CSV (ex: 'application_train.csv').
            
        Returns:
            pd.DataFrame: DataFrame com os dados carregados.
            
        Raises:
            DataLoadError: Se o arquivo não existir ou houver erro de leitura.
        """
        file_path = self.data_dir / filename
        
        if not file_path.is_file():
            error_msg = f"Arquivo não encontrado: {file_path}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)
            
        try:
            self.logger.info(f"Carregando dados de: {filename}...")
            # O engine 'pyarrow' pode acelerar consideravelmente a leitura de CSVs grandes
            df = pd.read_csv(file_path, engine="pyarrow") 
            self.logger.info(f"Sucesso! {filename} carregado com formato {df.shape}.")
            return df
        except Exception as e:
            error_msg = f"Erro ao ler o arquivo {filename}: {str(e)}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg)

    def load_application_train(self) -> pd.DataFrame:
        """Carrega a tabela principal de treinamento."""
        return self._load_csv("application_train.csv")

    def load_application_test(self) -> pd.DataFrame:
        """Carrega a tabela principal de teste."""
        return self._load_csv("application_test.csv")

    def load_bureau(self) -> pd.DataFrame:
        """Carrega os dados do bureau de crédito de outras instituições."""
        return self._load_csv("bureau.csv")

    def load_previous_applications(self) -> pd.DataFrame:
        """Carrega o histórico de aplicações anteriores no próprio Home Credit."""
        return self._load_csv("previous_application.csv")

    def load_bureau_balance(self) -> pd.DataFrame:
        """Carrega o histórico mensal de cada crédito externo (bureau_balance)."""
        return self._load_csv("bureau_balance.csv")

    def load_installments_payments(self) -> pd.DataFrame:
        """Carrega o histórico de pagamentos de parcelas de contratos anteriores."""
        return self._load_csv("installments_payments.csv")

    def load_pos_cash_balance(self) -> pd.DataFrame:
        """Carrega o saldo mensal de contratos POS/empréstimo pessoal anteriores."""
        return self._load_csv("POS_CASH_balance.csv")

    def load_credit_card_balance(self) -> pd.DataFrame:
        """Carrega o extrato mensal de cartões de crédito anteriores."""
        return self._load_csv("credit_card_balance.csv")

    def load_all_core_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Carrega as principais tabelas de uma só vez para análise ou feature engineering.

        Returns:
            Dict[str, pd.DataFrame]: Dicionário contendo os DataFrames.
        """
        self.logger.info("Iniciando carregamento em lote das tabelas principais...")

        return {
            "application_train": self.load_application_train(),
            "bureau": self.load_bureau(),
            "bureau_balance": self.load_bureau_balance(),
            "previous_application": self.load_previous_applications(),
            "installments_payments": self.load_installments_payments(),
            "pos_cash_balance": self.load_pos_cash_balance(),
            "credit_card_balance": self.load_credit_card_balance(),
        }