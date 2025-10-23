"""Shared utilities for debug tracing output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union
from uuid import UUID

__all__ = [
    "DebugLogWriter",
    "get_debug_log_writer",
    "prepare_for_logging",
]


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def prepare_for_logging(value: Any) -> Any:
    """Attempt to coerce ``value`` into a JSON-serialisable shape."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): prepare_for_logging(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [prepare_for_logging(v) for v in value]
    if hasattr(value, "dict"):
        try:
            return prepare_for_logging(value.dict())
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    if hasattr(value, "model_dump_json"):
        try:
            return json.loads(value.model_dump_json())
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    if hasattr(value, "model_dump"):
        try:
            return prepare_for_logging(value.model_dump())
        except Exception:  # pragma: no cover - defensive
            return repr(value)
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "message": str(value)}
    return repr(value)


class DebugLogWriter:
    """Write structured events into a single debug log file."""

    _instance_lock: Lock = Lock()
    _instance: Optional["DebugLogWriter"] = None

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._write_lock = Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_file()

    def _initialise_file(self) -> None:
        header = {
            "timestamp": _timestamp(),
            "event": "debug_log_start",
            "log_path": str(self.log_path),
        }
        with self._write_lock:
            with self.log_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(header) + "\n")

    def log(self, event: str, payload: Optional[dict[str, Any]] = None) -> None:
        record = {
            "timestamp": _timestamp(),
            "event": event,
            "payload": prepare_for_logging(payload or {}),
        }
        with self._write_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    @classmethod
    def get_instance(
        cls, log_path: Optional[Union[str, Path]] = None
    ) -> "DebugLogWriter":
        with cls._instance_lock:
            if cls._instance is None:
                path = Path(log_path) if log_path else cls._default_path()
                cls._instance = DebugLogWriter(path)
            elif log_path is not None:
                path = Path(log_path)
                if cls._instance.log_path != path:
                    cls._instance = DebugLogWriter(path)
            return cls._instance

    @staticmethod
    def _default_path() -> Path:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"azure_ai_debug_trace_{timestamp}.log"
        return Path(__file__).resolve().parent / filename


def get_debug_log_writer(
    log_path: Optional[Union[str, Path]] = None
) -> DebugLogWriter:
    """Return a singleton writer initialised for the given path."""

    return DebugLogWriter.get_instance(log_path)
