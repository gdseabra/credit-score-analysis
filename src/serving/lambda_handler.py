"""
Entrypoint AWS Lambda — adapta a aplicação FastAPI para o runtime do Lambda.

O Mangum traduz eventos do API Gateway (HTTP API, payload format 2.0) em chamadas
ASGI. ``lifespan="on"`` garante que o evento de startup do FastAPI rode no cold
start, disparando ``load_model()`` (que carrega o modelo do registro S3).
"""

from mangum import Mangum

from src.api.main import app

handler = Mangum(app, lifespan="on")
