"""Private, project-scoped MCP credentials for embedded Hermes sessions."""

from __future__ import annotations

import hashlib
import os
import secrets
from pathlib import Path

import portalocker
from sqlalchemy.exc import IntegrityError

from lib.db import async_session_factory
from lib.db.repositories.api_key_repository import ApiKeyRepository
from lib.infra.async_thread import run_sync_transaction
from lib.infra.data_root_layout import PROJECT_NAME_PATTERN, DataRootLayout
from server.auth import API_KEY_PREFIX, hash_api_key

SCOPED_KEY_NAME_PREFIX = "internal-hermes-project:"


def _private_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with portalocker.Lock(str(path.with_suffix(".lock")), timeout=10):
        if path.exists():
            if path.is_symlink():
                raise ValueError("Hermes credential file must be a regular private file")
            token = path.read_text(encoding="utf-8").strip()
            if not token.startswith(API_KEY_PREFIX) or len(token) < 40:
                raise ValueError("Hermes credential file is invalid")
            return token
        token = API_KEY_PREFIX + secrets.token_urlsafe(32)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(token)
        return token


async def ensure_project_mcp_token(project_name: str) -> str:
    """Provision one credential for the exact project without exposing it in API responses."""
    if not PROJECT_NAME_PATTERN.fullmatch(project_name):
        raise ValueError("Invalid Hermes project scope")
    filename = hashlib.sha256(project_name.encode()).hexdigest() + ".token"
    path = DataRootLayout.current().hermes_runtime_dir / "credentials" / filename
    token = await run_sync_transaction(_private_token, path)
    key_hash = hash_api_key(token)
    async with async_session_factory() as session:
        repo = ApiKeyRepository(session)
        if await repo.get_by_hash(key_hash) is None:
            try:
                await repo.create(
                    name=SCOPED_KEY_NAME_PREFIX + project_name,
                    key_hash=key_hash,
                    key_prefix=token[:12],
                )
                await session.commit()
            except IntegrityError:
                await session.rollback()
                if await repo.get_by_hash(key_hash) is None:
                    raise
    return token


def project_scope_for_subject(subject: str) -> str | None:
    prefix = "apikey:" + SCOPED_KEY_NAME_PREFIX
    return subject[len(prefix) :] if subject.startswith(prefix) else None
