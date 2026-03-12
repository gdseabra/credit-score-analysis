"""
Camada de conexão com o banco de dados Postgres.

Fornece um engine SQLAlchemy configurado via variável de ambiente CREDIT_DB_CONN,
com criação automática do schema `credit_score` na primeira conexão.

Uso:
    from src.data.database import get_engine, CREDIT_SCHEMA

    df.to_sql("features", con=get_engine(), schema=CREDIT_SCHEMA, if_exists="replace")
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# String de conexão configurável por variável de ambiente.
# O valor padrão aponta para o Postgres do docker-compose.
CREDIT_DB_CONN: str = os.getenv(
    "CREDIT_DB_CONN",
    "postgresql+psycopg2://airflow:airflow@postgres/airflow",
)

CREDIT_SCHEMA: str = "credit_score"


def get_engine() -> Engine:
    """Retorna um Engine SQLAlchemy conectado ao Postgres.

    Na primeira chamada, cria o schema ``credit_score`` caso não exista.

    Returns:
        Engine SQLAlchemy pronto para uso com pandas ``to_sql`` / ``read_sql``.

    Raises:
        sqlalchemy.exc.OperationalError: Se o banco não estiver acessível.
    """
    engine = create_engine(CREDIT_DB_CONN, pool_pre_ping=True)

    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {CREDIT_SCHEMA}"))
        # SQLAlchemy 1.x (usado pelo Airflow) faz autocommit via execute()
        # SQLAlchemy 2.x requereria conn.commit() explícito

    return engine
