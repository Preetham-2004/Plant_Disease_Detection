import json
import os
import threading
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from hmac import compare_digest

from fastapi import HTTPException

from ..config import USER_STORE_PATH


_STORE_LOCK = threading.Lock()
_PBKDF2_ITERATIONS = 100_000


def _ensure_store() -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USER_STORE_PATH.exists():
        USER_STORE_PATH.write_text("[]", encoding="utf-8")


def _read_users() -> list[dict]:
    _ensure_store()
    try:
        return json.loads(USER_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="User store is corrupted.")


def _write_users(users: list[dict]) -> None:
    USER_STORE_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str, salt: bytes) -> str:
    return pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


def register_user(name: str, email: str, password: str) -> dict:
    normalized_email = _normalize_email(email)

    with _STORE_LOCK:
        users = _read_users()
        if any(user["email"] == normalized_email for user in users):
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        salt = os.urandom(16)
        user = {
            "name": name.strip(),
            "email": normalized_email,
            "salt": salt.hex(),
            "password_hash": _hash_password(password, salt),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        users.append(user)
        _write_users(users)

    return {"name": user["name"], "email": user["email"]}


def authenticate_user(email: str, password: str) -> dict:
    normalized_email = _normalize_email(email)

    with _STORE_LOCK:
        users = _read_users()

    user = next((item for item in users if item["email"] == normalized_email), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    salt = bytes.fromhex(user["salt"])
    candidate_hash = _hash_password(password, salt)
    if not compare_digest(candidate_hash, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    return {"name": user["name"], "email": user["email"]}
