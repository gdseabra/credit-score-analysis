"""
Autenticação JWT para a API REST de Credit Score.

Implementa:
- Hashing seguro de senhas com bcrypt (passlib).
- Geração e validação de tokens JWT (python-jose).
- Dependency FastAPI para proteção de endpoints (get_current_user).

Usuários de demonstração (troque em produção por banco de dados):
    admin   / admin123
    analyst / analyst123
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Configuração — em produção, carregue de variáveis de ambiente
# ---------------------------------------------------------------------------

SECRET_KEY: str = "credit-score-secret-key-TROQUE-EM-PRODUCAO-use-openssl-rand-hex-32"
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

# ---------------------------------------------------------------------------
# Contexto de hashing e esquema OAuth2
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ---------------------------------------------------------------------------
# Usuários em memória (substituir por banco de dados em produção)
# ---------------------------------------------------------------------------

_USERS_DB: dict[str, dict] = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
    },
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica se a senha em texto corresponde ao hash armazenado.

    Args:
        plain_password: Senha em texto enviada pelo usuário.
        hashed_password: Hash bcrypt armazenado.

    Returns:
        True se a senha for válida, False caso contrário.
    """
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Autentica o usuário verificando nome e senha.

    Args:
        username: Nome de usuário.
        password: Senha em texto.

    Returns:
        Dicionário do usuário se autenticado, None caso contrário.
    """
    user = _USERS_DB.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Gera um token JWT assinado com expiração.

    Args:
        data: Payload do token (normalmente {"sub": username}).
        expires_delta: Duração de validade do token.

    Returns:
        Token JWT como string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# Dependency FastAPI
# ---------------------------------------------------------------------------


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency que valida o token JWT e retorna o usuário autenticado.

    Injete nos endpoints protegidos via ``Depends(get_current_user)``.

    Args:
        token: Token Bearer extraído automaticamente do header Authorization.

    Returns:
        Dicionário do usuário autenticado.

    Raises:
        HTTPException 401: Se o token for inválido ou expirado.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido ou expirado. Faça login em POST /auth/token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = _USERS_DB.get(username)
    if user is None:
        raise credentials_exception
    return user
