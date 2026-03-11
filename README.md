# credit-score-analysis
├── data/               # CSVs originais (não versionados)
├── notebooks/          # EDA e experimentação
├── src/                # Código modular (Processamento, ML, API)
    ├── __init__.py
    ├── data/               # Camada de Dados (Ingestão e SQL)
    │   ├── __init__.py
    │   ├── loader.py       # Classes para carregar CSVs/SQL
    │   └── database.py     # Conexão com Banco de Dados (Postgres/Azure)
    ├── features/           # Engenharia de Features
    │   ├── __init__.py
    │   └── build_features.py # Transformações complexas (POO)
    ├── models/             # Machine Learning (Treino e Predição)
    │   ├── __init__.py
    │   ├── train.py        # Scripts de treinamento
    │   └── predict.py      # Lógica de inferência do modelo
    ├── api/                # Backend (FastAPI/Flask)
    │   ├── __init__.py
    │   ├── main.py         # Entrypoint da API
    │   ├── routes.py       # Definição dos endpoints
    │   └── schemas.py      # Validação de dados (Pydantic)
    └── utils/              # Funções utilitárias e IA Generativa
        ├── __init__.py
        ├── logger.py       # Configuração de logs
        └── genai_client.py # Integração com LLMs (OpenAI/Azure)
├── tests/              # Testes unitários
├── docker-compose.yml  # Orquestração (DB, Airflow, API)
└── requirements.txt