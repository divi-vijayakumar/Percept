"""Minimal stdlib .env loader (no dependency on python-dotenv).

Loads ``KEY=VALUE`` lines into ``os.environ`` *without* overriding values already set in the real
environment (real env wins). Used by the live lanes so ``make eval-live`` / ``make e2e`` pick up a
local ``.env`` the way the developer's shell would, without a sourcing step.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: str | Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) looking for a ``.env`` file."""
    here = Path(start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Load a ``.env`` into ``os.environ`` (real env wins). Returns the var names set."""
    env_path = Path(path) if path else find_dotenv()
    if env_path is None or not env_path.exists():
        return []
    loaded: list[str] = []
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key not in os.environ:  # real environment takes precedence
            os.environ[key] = value
            loaded.append(key)
    return loaded


__all__ = ["find_dotenv", "load_dotenv"]
