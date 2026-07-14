"""
DocuMind Auth — Security Utilities
------------------------------------
Password hashing (bcrypt via passlib) and JWT access token
creation/verification (python-jose).
"""
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from src.utils.config import get_settings
from src.utils.exceptions import AuthenticationError

settings = get_settings()


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage."""
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against its stored hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(subject: str) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: the value stored in the token's "sub" claim — we use
                 the user's UUID (as a string) so the token identifies
                 exactly one user.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.auth.jwt_access_token_expire_minutes
    )
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(
        to_encode, settings.auth.jwt_secret_key, algorithm=settings.auth.jwt_algorithm
    )


def decode_access_token(token: str) -> str:
    """
    Decode and validate a JWT access token, returning the subject (user ID).
    Raises AuthenticationError on any invalid/expired/malformed token.
    """
    try:
        payload = jwt.decode(
            token, settings.auth.jwt_secret_key, algorithms=[settings.auth.jwt_algorithm]
        )
        subject: str | None = payload.get("sub")
        if subject is None:
            raise AuthenticationError("Token missing subject claim.")
        return subject
    except JWTError as exc:
        raise AuthenticationError("Invalid or expired token.") from exc