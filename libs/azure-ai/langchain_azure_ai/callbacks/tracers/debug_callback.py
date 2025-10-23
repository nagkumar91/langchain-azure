"""Callback handler that logs every LangChain lifecycle event."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from .debug_log_writer import get_debug_log_writer, prepare_for_logging

__all__ = [
    "DebuggingCallbackHandler",
]


class DebuggingCallbackHandler(BaseCallbackHandler):
    """Emit detailed callback information to a debug log file."""

    def __init__(
        self,
        log_path: Optional[str] = None,
        *,
        include_inputs: bool = True,
        include_outputs: bool = True,
        include_metadata: bool = True,
        log_new_tokens: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._logger = get_debug_log_writer(log_path)
        self._include_inputs = include_inputs
        self._include_outputs = include_outputs
        self._include_metadata = include_metadata
        self._log_new_tokens = log_new_tokens
        self._name = name or self.__class__.__name__
        self._logger.log(
            "debug_callback_initialised",
            {
                "name": self._name,
                "include_inputs": include_inputs,
                "include_outputs": include_outputs,
                "include_metadata": include_metadata,
                "log_new_tokens": log_new_tokens,
            },
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id, tags=tags, metadata=metadata)
        payload["serialized"] = prepare_for_logging(serialized)
        if self._include_inputs:
            payload["inputs"] = prepare_for_logging(inputs)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_chain_start", payload)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        if self._include_outputs:
            payload["outputs"] = prepare_for_logging(outputs)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_chain_end", payload)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        payload["error"] = prepare_for_logging(error)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_chain_error", payload)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id, tags=tags, metadata=metadata)
        payload["serialized"] = prepare_for_logging(serialized)
        if self._include_inputs:
            payload["prompts"] = prepare_for_logging(prompts)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_llm_start", payload)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        if self._include_outputs:
            payload["response"] = prepare_for_logging(response)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_llm_end", payload)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        payload["error"] = prepare_for_logging(error)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_llm_error", payload)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id, tags=tags, metadata=metadata)
        payload["serialized"] = prepare_for_logging(serialized)
        if self._include_inputs:
            payload["messages"] = prepare_for_logging(messages)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_chat_model_start", payload)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self._log_new_tokens:
            return None
        payload = self._base_payload(run_id, parent_run_id)
        payload["token"] = token
        if chunk is not None:
            payload["chunk"] = prepare_for_logging(chunk)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_llm_new_token", payload)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id, tags=tags, metadata=metadata)
        payload["serialized"] = prepare_for_logging(serialized)
        if self._include_inputs:
            payload["input_str"] = input_str
            if inputs is not None:
                payload["inputs"] = prepare_for_logging(inputs)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_tool_start", payload)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        if self._include_outputs:
            payload["output"] = prepare_for_logging(output)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_tool_end", payload)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        payload["error"] = prepare_for_logging(error)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_tool_error", payload)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        payload["action"] = prepare_for_logging(action)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_agent_action", payload)

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        if self._include_outputs:
            payload["finish"] = prepare_for_logging(finish)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_agent_finish", payload)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id, tags=tags, metadata=metadata)
        payload["serialized"] = prepare_for_logging(serialized)
        if self._include_inputs:
            payload["query"] = query
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_retriever_start", payload)

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        if self._include_outputs:
            payload["documents"] = prepare_for_logging(documents)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_retriever_end", payload)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        payload["error"] = prepare_for_logging(error)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_retriever_error", payload)

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        if self._include_outputs:
            payload["text"] = text
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_text", payload)

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, parent_run_id)
        payload["retry_state"] = prepare_for_logging(retry_state)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_retry", payload)

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = self._base_payload(run_id, None, tags=tags, metadata=metadata)
        payload["name"] = name
        payload["data"] = prepare_for_logging(data)
        if kwargs:
            payload["extra_kwargs"] = prepare_for_logging(kwargs)
        self._logger.log("on_custom_event", payload)

    def _base_payload(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "callback": self._name,
            "run_id": str(run_id),
        }
        if parent_run_id:
            payload["parent_run_id"] = str(parent_run_id)
        if self._include_metadata:
            if tags:
                payload["tags"] = prepare_for_logging(tags)
            if metadata:
                payload["metadata"] = prepare_for_logging(metadata)
        return payload
