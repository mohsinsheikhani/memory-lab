"""Who is asking. The tenant and user come from a verified JWT, never from the model."""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

load_dotenv()

ALGORITHM = "HS256"

_scope: ContextVar[tuple[str, str] | None] = ContextVar("scope", default=None)


def make_token(tenant_id: str, user_id: str, hours: int = 1) -> str:
    """Stands in for the login server."""
    now = datetime.now(timezone.utc)
    claims = {"tenant_id": tenant_id, "sub": user_id, "iat": now, "exp": now + timedelta(hours=hours)}
    return jwt.encode(claims, os.environ["JWT_SECRET"], algorithm=ALGORITHM)


@contextmanager
def login(token: str):
    """Verify the token and set the scope for everything inside the block."""
    claims = jwt.decode(token, os.environ["JWT_SECRET"], algorithms=[ALGORITHM])  # raises if forged or expired
    reset = _scope.set((claims["tenant_id"], claims["sub"]))
    try:
        yield
    finally:
        _scope.reset(reset)


def current_scope() -> tuple[str, str]:
    scope = _scope.get()
    if scope is None:
        raise PermissionError("no logged-in user")
    return scope


if __name__ == "__main__":
    token = make_token("freshcart", "ali")
    print(token)
    with login(token):
        print("scope:", current_scope())

    try:
        login(token[:-2] + "xx").__enter__()
    except jwt.InvalidTokenError as e:
        print("forged token:", type(e).__name__)
