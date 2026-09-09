import os
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SESSION_COOKIE = "mendeley_admin_session"
SESSION_MAX_AGE = 60 * 60 * 12


def _serializer() -> URLSafeTimedSerializer:
    secret = os.getenv("ADMIN_PASSWORD", "change-me")
    return URLSafeTimedSerializer(secret, salt="mendeley-admin")


def create_session_token() -> str:
    return _serializer().dumps({"role": "admin"})


def verify_password(password: str) -> bool:
    expected = os.getenv("ADMIN_PASSWORD", "change-me")
    return password == expected


def require_admin(
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> None:
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    try:
        _serializer().loads(session, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired.")
