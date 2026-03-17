# Credit Score Analysis

Sistema de análise e predição de risco de crédito com explicabilidade por IA, construído com dados do [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle).

**App em produção:** [https://front-end-production-3c12.up.railway.app](https://front-end-production-3c12.up.railway.app)

---

## Visão Geral

O sistema avalia a probabilidade de inadimplência de um solicitante de crédito e entrega:

- **Decisão** (APROVADO / NEGADO) com probabilidade e faixa de risco (A–E)
- **Explicação das features mais importantes** via SHAP values
- **Narrativa em português** gerada pelo Claude (Anthropic) contextualizada nas features mais relevantes do perfil do cliente

### Arquitetura

```
┌─────────────────┐    HTTP     ┌──────────────────────────────────────────┐
│   Streamlit UI  │ ──────────▶ │  FastAPI (REST)                          │
│  (Frontend)     │             │  ├── POST /predict/                       │
└─────────────────┘             │  ├── POST /predict/batch                  │
                                │  ├── POST /explain/  ◀── SHAP + Claude    │
                                │  ├── POST /auth/token                     │
                                │  └── GET  /model/info                     │
                                └─────────────────┬────────────────────────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          │                       │                       │
                   ┌──────▼──────┐       ┌────────▼───────┐    ┌─────────▼──────┐
                   │  LightGBM   │       │  PostgreSQL 16  │    │  MLflow Server │
                   │  Pipeline   │       │  (features +    │    │  (experimentos │
                   │  (.joblib)  │       │   histórico)    │    │   e métricas)  │
                   └─────────────┘       └────────────────┘    └────────────────┘

         Airflow DAG (credit_score_etl): extract → transform → load → train_model
```

---

## Stack de Tecnologia

| Camada | Tecnologia |
|---|---|
| **Frontend** | Streamlit 1.35 |
| **Backend API** | FastAPI 0.111, Uvicorn 0.30 |
| **Autenticação** | JWT (python-jose), bcrypt (passlib) |
| **ML Principal** | LightGBM 4.3 |
| **ML Alternativo** | XGBoost 2.0, Random Forest, Logistic Regression |
| **AutoML** | FLAML 2.1 |
| **Deep Learning** | PyTorch 2.10 |
| **Explainability** | SHAP 0.45 (TreeExplainer) + Claude Haiku 4.5 (Anthropic) |
| **Feature Engineering** | scikit-learn 1.5 (Pipeline, ColumnTransformer) |
| **Dados** | pandas 2.2, numpy 1.26, pyarrow 15 |
| **Banco de Dados** | PostgreSQL 16, SQLAlchemy 2.0, psycopg2 2.9 |
| **Orquestração** | Apache Airflow 2.9 |
| **Tracking de Experimentos** | MLflow 2.14 |
| **Containers** | Docker, Docker Compose |
| **Testes** | pytest |

---

## Estrutura do Projeto

```
credit-score-analysis/
├── dags/
│   └── credit_score_etl.py          # DAG Airflow: ETL + treinamento
├── data/
│   ├── application_train.csv        # Dataset bruto (Home Credit)
│   ├── application_test.csv
│   ├── bureau.csv, bureau_balance.csv, ...
│   ├── interim/                     # Parquets intermediários
│   ├── processed/                   # Parquets finais
│   └── models/                      # Modelos serializados (.joblib)
├── notebooks/                       # EDA e experimentação (6 notebooks)
├── src/
│   ├── api/
│   │   ├── main.py                  # FastAPI entrypoint
│   │   ├── auth.py                  # Lógica JWT
│   │   ├── dependencies.py          # Injeção de dependências (carrega modelo)
│   │   ├── schemas.py               # Pydantic models (request/response)
│   │   └── routes/
│   │       ├── predict.py           # /predict/ e /predict/batch
│   │       ├── explain.py           # /explain/ (SHAP + LLM)
│   │       ├── auth.py              # /auth/token
│   │       └── model.py             # /model/info
│   ├── data/
│   │   ├── loader.py                # HomeCreditDataLoader
│   │   └── database.py              # Conexão PostgreSQL
│   ├── features/
│   │   └── build_features.py        # AnomalyHandler, DomainFeatureBuilder
│   ├── models/
│   │   ├── classifiers.py           # Factory: LightGBM / XGBoost / RF / LR
│   │   ├── trainer.py               # CreditTrainer (cross-val + MLflow)
│   │   ├── evaluator.py             # Métricas: AUC, Gini, KS, F1
│   │   ├── automl.py                # AutoML via FLAML
│   │   ├── clustering.py            # Segmentação de clientes
│   │   └── deep_learning.py         # Modelos PyTorch
│   ├── explainability/
│   │   ├── shap_explainer.py        # Wrapper SHAP TreeExplainer
│   │   ├── llm_explainer.py         # Integração Claude (Anthropic)
│   │   └── knowledge_base.py        # Base de conhecimento (RAG)
│   ├── analysis/
│   │   ├── descriptive.py
│   │   ├── hypothesis.py
│   │   └── visualization.py
│   └── frontend/
│       ├── app.py                   # Streamlit app principal
│       ├── api_client.py            # HTTP client para a API
│       └── components/
│           ├── input_form.py        # Formulário de solicitação
│           ├── result_card.py       # Card de resultado / decisão
│           └── explanation_panel.py # Painel SHAP + explicação IA
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   └── test_loader.py
├── docker-compose.yml               # Stack completa (dev)
├── docker-compose.prod.yml          # Stack de produção (API + Frontend)
├── Dockerfile.api
├── Dockerfile.frontend
├── Dockerfile.airflow
├── requirements.txt                 # Dependências completas (com Airflow)
└── requirements.api.txt             # Dependências mínimas da API
```

---

## Modelo de Machine Learning

### Pipeline de Treinamento

O treinamento é orquestrado pelo DAG `credit_score_etl` no Airflow em 4 etapas:

1. **Extract** — Carrega CSVs brutos do Home Credit (`application_train.csv` + tabelas auxiliares)
2. **Transform** — Aplica engenharia de features:
   - `AnomalyHandler`: Trata valor sentinela `DAYS_EMPLOYED=365243` (sem emprego) → cria flag + substitui por NaN
   - `DomainFeatureBuilder`: Cria features derivadas (ratios, conversão de dias → anos)
3. **Load** — Persiste dados processados no PostgreSQL (`credit_score.features`)
4. **Train** — Treina `LightGBM` com validação cruzada estratificada (5-fold), registra métricas no MLflow e salva o pipeline em `models/lightgbm_pipeline.joblib`

### Features Utilizadas (24 total)

**Numéricas (16)**
| Feature | Descrição |
|---|---|
| `AMT_CREDIT` | Valor do crédito solicitado |
| `AMT_ANNUITY` | Valor da prestação anual |
| `AMT_INCOME_TOTAL` | Renda total do cliente |
| `AMT_GOODS_PRICE` | Preço do bem financiado |
| `DAYS_BIRTH` | Idade em dias (negativo) |
| `DAYS_EMPLOYED` | Tempo de emprego em dias (negativo) |
| `DAYS_REGISTRATION` | Dias desde registro (negativo) |
| `DAYS_ID_PUBLISH` | Dias desde emissão do documento (negativo) |
| `CNT_FAM_MEMBERS` | Número de membros na família |
| `CREDIT_INCOME_RATIO` | Crédito / Renda (engineered) |
| `ANNUITY_INCOME_RATIO` | Prestação / Renda (engineered) |
| `CREDIT_TERM_MONTHS` | Prazo do crédito em meses (engineered) |
| `AGE_YEARS` | Idade em anos (engineered) |
| `EMPLOYED_YEARS` | Anos de emprego (engineered) |
| `EMPLOYED_TO_AGE_RATIO` | Anos empregado / Idade (engineered) |
| `INCOME_PER_FAMILY_MEMBER` | Renda per capita familiar (engineered) |

**Categóricas (8)**
`CODE_GENDER`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`

### Métricas de Avaliação

Rastreadas no MLflow a cada run de treinamento:
- **AUC-ROC** (média ± desvio padrão — 5-fold CV)
- **Gini** (= 2 × AUC − 1)
- **KS** (Kolmogorov-Smirnov)
- **F1, Precision, Recall**

### Outros Modelos Disponíveis

A factory `CreditClassifier.get_model()` suporta: `lightgbm` (padrão), `xgboost`, `random_forest`, `logistic_regression`.

---

## API REST

**Base URL local:** `http://localhost:8000`
**Swagger UI:** `http://localhost:8000/docs`
**ReDoc:** `http://localhost:8000/redoc`

### Endpoints

#### `GET /health`
Verifica se a API e o modelo estão disponíveis. Retorna HTTP 503 se o modelo não estiver carregado.

#### `POST /auth/token`
Retorna um JWT Bearer token (válido por 30 minutos).

```json
// Request
{ "username": "admin", "password": "admin123" }

// Response
{ "access_token": "...", "token_type": "bearer", "expires_in": 1800 }
```

Usuários de demonstração: `admin/admin123`, `analyst/analyst123`.

#### `POST /predict/` _(requer autenticação)_
Predição individual. Retorna probabilidade de inadimplência, decisão e faixa de risco.

```json
// Response
{
  "probability_default": 0.23,
  "decision": "APROVADO",
  "score_band": "B",
  "threshold": 0.5
}
```

**Faixas de risco:** A (< 20%) · B (20–40%) · C (40–60%) · D (60–80%) · E (> 80%)

#### `POST /predict/batch` _(requer autenticação)_
Predição em lote (até 1.000 solicitações por requisição).

#### `POST /explain/` _(requer autenticação)_
Predição + SHAP values + narrativa gerada pelo Claude em português.

```json
// Response
{
  "probability_default": 0.23,
  "decision": "APROVADO",
  "score_band": "B",
  "threshold": 0.5,
  "shap_features": [
    {
      "name": "CREDIT_INCOME_RATIO",
      "label": "Proporção crédito/renda",
      "shap_value": 0.15,
      "feature_value": 3.2,
      "description": "Relação entre o valor do crédito e a renda anual..."
    }
  ],
  "explanation": "Com base no perfil analisado, o principal fator de risco identificado foi..."
}
```

#### `GET /model/info` _(requer autenticação)_
Retorna metadados do modelo carregado (nome, caminho, features, status).

---

## Explicabilidade (RAG + SHAP + Claude)

O endpoint `/explain/` implementa um padrão RAG (_Retrieval-Augmented Generation_):

1. **Retrieve** — SHAP `TreeExplainer` calcula a contribuição de cada feature para a predição (nativo para LightGBM, ~10× mais rápido que KernelExplainer)
2. **Augment** — A `knowledge_base.py` enriquece cada feature com descrição de negócio, rótulo amigável e direção de risco (alto/baixo)
3. **Generate** — O Claude Haiku 4.5 recebe as top-5 features + contexto e gera uma narrativa empática em português, sem jargão técnico

---

## Rodando Localmente

### Pré-requisitos

- Docker e Docker Compose instalados
- Arquivo `.env` na raiz com:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

### Opção 1 — Stack completa (recomendado para desenvolvimento)

Sobe PostgreSQL, Airflow, MLflow, API e Frontend:

```bash
docker-compose up --build
```

| Serviço | URL | Credenciais |
|---|---|---|
| Frontend (Streamlit) | http://localhost:8501 | admin / admin123 |
| API (Swagger) | http://localhost:8000/docs | — |
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | — |

**Após subir os containers, treine o modelo:**

1. Acesse http://localhost:8080 (Airflow)
2. Ative o DAG `credit_score_etl`
3. Dispare manualmente (**Trigger DAG**)
4. Aguarde todas as tasks: `extract_data → transform_data → load_to_db → train_model`
5. O modelo será salvo em `data/models/lightgbm_pipeline.joblib` e a API o carregará automaticamente

### Opção 2 — Apenas API + Frontend (produção local)

```bash
docker-compose -f docker-compose.prod.yml up --build
```

> Requer que o modelo já exista em `data/models/lightgbm_pipeline.joblib`.

### Opção 3 — Desenvolvimento sem Docker

```bash
# Instale as dependências
pip install -r requirements.api.txt

# Terminal 1: API
export PYTHONPATH=$(pwd)
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run src/frontend/app.py
```

---

## Testes

```bash
pytest tests/ -v
```

Os testes unitários usam DataFrames sintéticos (sem dependência dos CSVs originais) e cobrem `AnomalyHandler`, `DomainFeatureBuilder` e `HomeCreditDataLoader`.

---

## Variáveis de Ambiente

| Variável | Obrigatória | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | Sim (para `/explain/`) | Chave da API Claude (Anthropic) |
| `CREDIT_DB_CONN` | Sim (com DB) | URL SQLAlchemy do PostgreSQL |
| `MLFLOW_TRACKING_URI` | Não | URL do servidor MLflow (padrão: `http://mlflow:5000`) |
| `API_URL` | Não (frontend) | URL da API para o Streamlit (padrão: `http://api:8000`) |

---

## Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_analise_exploratoria.ipynb` | EDA inicial do dataset Home Credit |
| `02_feature_engineering.ipynb` | Desenvolvimento e validação das features |
| `03_validacao_banco.ipynb` | Verificação dos dados carregados no PostgreSQL |
| `04_analise_exploratoria_features.ipynb` | Análise aprofundada das features engineered |
| `05_testes_modelos_ml.ipynb` | Comparação de modelos e tuning de hiperparâmetros |
| `data_loader.ipynb` | Exploração da camada de carregamento de dados |

---

## Dataset

**Home Credit Default Risk** — Kaggle Competition
O dataset contém informações de solicitações de crédito com a variável alvo `TARGET` (1 = inadimplente, 0 = adimplente). Arquivos principais: `application_train.csv`, `application_test.csv` e tabelas auxiliares (`bureau`, `previous_application`, etc.).

> Os CSVs brutos não são versionados no repositório. Faça o download em [kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk) e coloque em `data/`.
