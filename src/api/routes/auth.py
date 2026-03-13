"""
Rota de autenticação — POST /auth/token.

Recebe username/password e devolve um token JWT Bearer
que deve ser enviado no header ``Authorization: Bearer <token>``
nas demais requisições.
"""

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status

from src.api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
)
from src.api.schemas import TokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Autenticação"])


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Obter token de acesso",
    description=(
        "Autentica o usuário com username e password e retorna um token JWT.\n\n"
        "**Usuários de demo:** `admin / admin123` ou `analyst / analyst123`"
    ),
)
def login(credentials: TokenRequest) -> TokenResponse:
    """Endpoint de login — gera token JWT para acesso à API.

    Args:
        credentials: Usuário e senha em JSON.

    Returns:
        Token JWT com tipo e expiração.

    Raises:
        HTTPException 401: Se as credenciais forem inválidas.
    """
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
