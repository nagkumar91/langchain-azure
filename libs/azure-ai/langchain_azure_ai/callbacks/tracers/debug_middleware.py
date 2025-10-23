"""Middleware that writes detailed execution traces into a debug log."""

from __future__ import annotations

from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware

from .debug_log_writer import get_debug_log_writer, prepare_for_logging

__all__ = [
    "DebuggingAgentMiddleware",
]


class DebuggingAgentMiddleware(AgentMiddleware):
    """Agent middleware that logs every hook invocation to a file."""

    def __init__(
        self,
        log_path: Optional[str] = None,
        *,
        include_state_snapshot: bool = True,
        include_runtime_snapshot: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self._logger = get_debug_log_writer(log_path)
        self._include_state = include_state_snapshot
        self._include_runtime = include_runtime_snapshot
        self._custom_name = name or self.__class__.__name__
        self._logger.log(
            "debug_middleware_initialised",
            {
                "name": self._custom_name,
                "include_state_snapshot": include_state_snapshot,
                "include_runtime_snapshot": include_runtime_snapshot,
            },
        )

    @property
    def name(self) -> str:
        return self._custom_name

    def before_agent(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("before_agent", state=state, runtime=runtime)
        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("abefore_agent", state=state, runtime=runtime)
        return None

    def before_model(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("before_model", state=state, runtime=runtime)
        return None

    async def abefore_model(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("abefore_model", state=state, runtime=runtime)
        return None

    def after_model(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("after_model", state=state, runtime=runtime)
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("aafter_model", state=state, runtime=runtime)
        return None

    def after_agent(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("after_agent", state=state, runtime=runtime)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        self._log("aafter_agent", state=state, runtime=runtime)
        return None

    def _log(self, hook: str, **items: Any) -> None:
        payload = {
            "hook": hook,
            "middleware": self._custom_name,
        }
        if self._include_state and "state" in items:
            payload["state"] = prepare_for_logging(items["state"])
        if self._include_runtime and "runtime" in items:
            payload["runtime"] = prepare_for_logging(items["runtime"])
        self._logger.log("middleware_hook", payload)
