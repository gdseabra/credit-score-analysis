"""
API REST de predição de risco de crédito — Credit Score API.

Stack:
    FastAPI + Uvicorn + Pydantic v2

Endpoints públicos (sem autenticação):
    GET  /health          — Health check da aplicação e status do modelo.

Endpoints protegidos (requerem JWT):
    POST /auth/token      — Login e obtenção de token JWT.
    POST /predict/        — Predição individual de risco de crédito.
    POST /predict/batch   — Predição em lote (até 1.000 solicitações).
    GET  /model/info      — Metadados e status do modelo carregado.

Documentação interativa:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

Uso local:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Uso via Docker:
    docker-compose up api
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import is_model_loaded, load_model
from src.api.routes import auth, model, predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação.

    No startup: tenta carregar o pipeline treinado em memória.
    No shutdown: libera recursos (logging).
    """
    logger.info("Iniciando Credit Score API...")
    load_model()
    if is_model_loaded():
        logger.info("Pipeline ML carregado com sucesso.")
    else:
        logger.warning(
            "Nenhum modelo encontrado. Endpoints de predição retornarão 503 "
            "até que o DAG 'credit_score_etl' seja executado no Airflow."
        )
    yield
    logger.info("Encerrando Credit Score API.")


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Score API",
    description=(
        "API REST para predição de risco de crédito baseada no dataset "
        "[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).\n\n"
        "## Autenticação\n"
        "Todos os endpoints de predição exigem um token JWT.\n"
        "1. Faça `POST /auth/token` com `username` e `password`.\n"
        "2. Copie o `access_token` retornado.\n"
        "3. Clique em **Authorize** (cadeado) e cole o token.\n\n"
        "**Usuários de demo:** `admin / admin123` · `analyst / analyst123`\n\n"
        "## Modelo\n"
        "Pipeline LightGBM treinado com validação cruzada estratificada (5-fold). "
        "Métricas rastreadas no MLflow (`http://localhost:5000`)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Credit Score Project",
        "url": "https://github.com/gabrielseabra/credit-score-analysis",
    },
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(model.router)


# ---------------------------------------------------------------------------
# Endpoints utilitários
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    tags=["Sistema"],
    summary="Health check",
    description="Verifica se a API está operacional e se o modelo está carregado.",
)
def health() -> JSONResponse:
    """Health check sem autenticação.

    Returns:
        JSON com status da API e disponibilidade do modelo.
        HTTP 200 se saudável, HTTP 503 se o modelo não estiver carregado.
    """
    model_loaded = is_model_loaded()
    body = {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_status": "disponível" if model_loaded else "indisponível",
        "docs": "/docs",
    }
    http_status = 200 if model_loaded else 503
    return JSONResponse(content=body, status_code=http_status)
