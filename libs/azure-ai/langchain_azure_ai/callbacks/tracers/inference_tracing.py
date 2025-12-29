"""OpenTelemetry tracer for LangChain/LangGraph inference aligned with GenAI spec.

This implementation emits spans for agent execution, model invocations, tool
calls, and retriever activity and records attributes in line with the
OpenTelemetry GenAI semantic conventions.  It supports simultaneous export to
Azure Monitor (when a connection string is supplied) and to any OTLP collector
configured via environment variables (for example::

    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export OTEL_EXPORTER_OTLP_PROTOCOL=grpc

The tracer focuses on three design goals:

1.  Emit spec-compliant spans with the required/conditional GenAI attributes.
2.  Capture as much context as LangChain exposes (messages, tool schemas,
    model parameters, token usage) while allowing content redaction.
3.  Provide safe defaults that work across Azure OpenAI, public OpenAI,
    GitHub Models, Ollama, and other OpenAI-compatible deployments.

The module exports the ``AzureAIOpenTelemetryTracer`` callback handler.  Attach
an instance to LangChain run configs (for example in ``config["callbacks"]``)
to instrument your applications.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union, cast
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

try:  # pragma: no cover - imported lazily in production environments
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import attach, detach
    from opentelemetry.propagate import extract
    from opentelemetry.semconv.schemas import Schemas
    from opentelemetry.trace import (
        Span,
        SpanKind,
        Status,
        StatusCode,
        get_current_span,
        set_span_in_context,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Azure OpenTelemetry tracing requires 'azure-monitor-opentelemetry' "
        "and 'opentelemetry-sdk'. Install them via:\n"
        "    pip install azure-monitor-opentelemetry opentelemetry-sdk"
    ) from exc

LOGGER = logging.getLogger(__name__)
_DEBUG_THREAD_STACK = os.getenv("AZURE_TRACING_DEBUG_THREAD_STACK", "").lower() in {
    "1",
    "true",
    "yes",
}
if _DEBUG_THREAD_STACK:
    logging.basicConfig(level=logging.INFO)

_LANGGRAPH_GENERIC_NAME = "LangGraph"
_LANGGRAPH_START_NODE = "__start__"
_LANGGRAPH_MIDDLEWARE_PREFIX = "Middleware."
_LANGCHAIN_TRACER_CONTEXT_VAR: ContextVar[BaseCallbackHandler | None] = ContextVar(
    "azure_ai_genai_tracer_callback",
    default=None,
)

# ---------------------------------------------------------------------------
# Default configuration for static autolog() API
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "provider_name": None,  # Auto-infer if None
    "tracer_name": "azure_ai_genai_tracer",
    "enable_content_recording": True,
    "redact_messages": False,
    "redact_tool_arguments": False,
    "redact_tool_results": False,
    "include_tool_definitions": True,
    "aggregate_usage": True,
    "normalize_operation_names": True,
    "sampling_rate": 1.0,
    "honor_external_parent": True,
    "patch_mode": "callback",  # "callback", "hybrid", or "monkey"
    "log_stream_chunks": False,
    "max_message_characters": None,  # Optional truncation
    "thread_id_attribute": "gen_ai.conversation.id",
    "enable_span_links": True,
    "message_keys": ("messages",),
    "message_paths": (),
    "trace_all_langgraph_nodes": False,
    "ignore_start_node": True,
    "compat_create_agent_filtering": True,
    "enable_performance_counters": None,
}


class Attrs:
    """Semantic convention attribute names used throughout the tracer."""

    PROVIDER_NAME = "gen_ai.provider.name"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_MAX_INPUT_TOKENS = "gen_ai.request.max_input_tokens"
    REQUEST_MAX_OUTPUT_TOKENS = "gen_ai.request.max_output_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_STOP = "gen_ai.request.stop_sequences"
    REQUEST_FREQ_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRES_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    REQUEST_SEED = "gen_ai.request.seed"
    REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    OUTPUT_TYPE = "gen_ai.output.type"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    DATA_SOURCE_ID = "gen_ai.data_source.id"
    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    CONVERSATION_ID = "gen_ai.conversation.id"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    ERROR_TYPE = "error.type"
    RETRIEVER_RESULTS = "gen_ai.retriever.results"
    RETRIEVER_QUERY = "gen_ai.retriever.query"

    # Optional vendor-specific attributes
    OPENAI_REQUEST_SERVICE_TIER = "openai.request.service_tier"
    OPENAI_RESPONSE_SERVICE_TIER = "openai.response.service_tier"
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "openai.response.system_fingerprint"


def _as_json_attribute(value: Any) -> str:
    """Return a JSON string suitable for OpenTelemetry string attributes."""
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive
        return json.dumps(str(value), ensure_ascii=False)


def _redact_text_content() -> dict[str, str]:
    return {"type": "text", "content": "[redacted]"}


def _message_role(message: Union[BaseMessage, dict[str, Any]]) -> str:
    if isinstance(message, BaseMessage):
        # LangChain message types map to GenAI roles
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, ToolMessage):
            return "tool"
        if isinstance(message, AIMessage):
            return "assistant"
        return message.type
    role = message.get("role") or message.get("type")
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    if role == "tool":
        return "tool"
    if role == "system":
        return "system"
    return str(role or "user")


def _message_content(message: Union[BaseMessage, dict[str, Any]]) -> Any:
    if isinstance(message, BaseMessage):
        return message.content
    return message.get("content")


def _coerce_content_to_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(str(part) for part in content if part is not None)
    return str(content)


def _extract_tool_calls(
    message: Union[BaseMessage, dict[str, Any]],
) -> List[dict[str, Any]]:
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        tool_calls = getattr(message, "tool_calls") or []
        if isinstance(tool_calls, list):
            return [tc for tc in tool_calls if isinstance(tc, dict)]
    elif isinstance(message, dict):
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            return [tc for tc in tool_calls if isinstance(tc, dict)]
    return []


def _tool_call_id_from_message(
    message: Union[BaseMessage, dict[str, Any]],
) -> Optional[str]:
    if isinstance(message, ToolMessage):
        if getattr(message, "tool_call_id", None):
            return str(message.tool_call_id)
    if isinstance(message, dict):
        if message.get("tool_call_id"):
            return str(message["tool_call_id"])
    return None


def _prepare_messages(
    raw_messages: Any,
    *,
    record_content: bool,
    include_roles: Optional[Iterable[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return (formatted_messages_json, system_instructions_json)."""
    if not raw_messages:
        return None, None

    include_role_set = set(include_roles) if include_roles is not None else None

    if isinstance(raw_messages, dict):
        iterable: Sequence[Any] = raw_messages.get("messages") or []
    elif isinstance(raw_messages, (list, tuple)):
        if raw_messages and isinstance(raw_messages[0], (list, tuple)):
            iterable = [msg for thread in raw_messages for msg in thread]
        else:
            iterable = list(raw_messages)
    else:
        iterable = [raw_messages]

    formatted: List[dict[str, Any]] = []
    system_parts: List[dict[str, str]] = []

    for item in iterable:
        role = _message_role(item)
        content = _coerce_content_to_text(_message_content(item))

        if role == "system":
            if content:
                system_parts.append(
                    {
                        "type": "text",
                        "content": content if record_content else "[redacted]",
                    }
                )
            continue

        if include_role_set is not None and role not in include_role_set:
            continue

        parts: List[dict[str, Any]] = []

        if role in {"user", "assistant"} and content:
            parts.append(
                {
                    "type": "text",
                    "content": content if record_content else "[redacted]",
                }
            )

        if role == "tool":
            tool_result = content if record_content else "[redacted]"
            parts.append(
                {
                    "type": "tool_call_response",
                    "id": _tool_call_id_from_message(item),
                    "result": tool_result,
                }
            )

        tool_calls = _extract_tool_calls(item)
        for tc in tool_calls:
            arguments = tc.get("args") or tc.get("arguments")
            if arguments is None:
                arguments = {}
            tc_entry = {
                "type": "tool_call",
                "id": tc.get("id"),
                "name": tc.get("name"),
                "arguments": arguments if record_content else "[redacted]",
            }
            parts.append(tc_entry)

        if not parts:
            parts.append(_redact_text_content())

        formatted.append({"role": role, "parts": parts})

    formatted_json = _as_json_attribute(formatted) if formatted else None
    system_json = _as_json_attribute(system_parts) if system_parts else None
    return formatted_json, system_json


def _filter_assistant_output(formatted_messages: str) -> Optional[str]:
    try:
        messages = json.loads(formatted_messages)
    except Exception:
        return formatted_messages
    cleaned: List[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text_parts = [
            part for part in msg.get("parts", []) if part.get("type") == "text"
        ]
        if not text_parts:
            continue
        cleaned.append({"role": "assistant", "parts": text_parts})
    if not cleaned:
        return None
    return _as_json_attribute(cleaned)


def _to_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:
            return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method(exclude_none=True)
        except TypeError:
            dumped = dict_method()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _unwrap_command_like(value: Any) -> tuple[Any, Optional[str]]:
    try:
        if hasattr(value, "update") and hasattr(value, "goto"):
            update = getattr(value, "update")
            goto = getattr(value, "goto")
            return update, str(goto) if goto else None
    except Exception:
        return value, None
    return value, None


def _extract_by_dot_path(container: Any, path: str) -> Any:
    current = container
    for segment in path.split("."):
        mapping = _to_mapping(current)
        if mapping is not None and segment in mapping:
            current = mapping[segment]
            continue
        if hasattr(current, segment):
            try:
                current = getattr(current, segment)
                continue
            except Exception:
                return None
        return None
    return current


def _effective_message_keys_paths(
    metadata: Mapping[str, Any] | None,
    default_keys: Sequence[str],
    default_paths: Sequence[str],
) -> tuple[Sequence[str], Sequence[str]]:
    keys: Sequence[str] = default_keys
    paths: Sequence[str] = default_paths
    if metadata and isinstance(metadata, Mapping):
        key_override = metadata.get("otel_messages_key")
        if isinstance(key_override, str):
            keys = (key_override,)
        else:
            keys_candidate = metadata.get("otel_messages_keys")
            try:
                if keys_candidate is not None and not isinstance(keys_candidate, str):
                    keys = tuple(str(k) for k in keys_candidate)
            except TypeError:
                keys = default_keys
        path_override = metadata.get("otel_messages_path")
        if isinstance(path_override, str):
            paths = (path_override,)
    return keys, paths


def _extract_messages_payload(
    container: Any, *, message_keys: Sequence[str], message_paths: Sequence[str]
) -> tuple[Any, Optional[str]]:
    unwrapped, goto = _unwrap_command_like(container)
    fallback_mapping: Optional[Mapping[str, Any]] = None
    for path in message_paths:
        value = _extract_by_dot_path(unwrapped, path)
        if value is not None:
            return value, goto
    mapping = _to_mapping(unwrapped)
    if mapping is not None:
        fallback_mapping = mapping
        for key in message_keys:
            if key in mapping:
                return mapping[key], goto
    for key in message_keys:
        if hasattr(unwrapped, key):
            try:
                return getattr(unwrapped, key), goto
            except Exception:
                continue
    if fallback_mapping is not None:
        return fallback_mapping, goto
    return unwrapped, goto


def _scrub_value(value: Any, record_content: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if not record_content:
            return "[redacted]"
        stripped = value.strip()
        if stripped and stripped[0] in "{[":
            try:
                return json.loads(stripped)
            except Exception:
                pass
        return value
    if not record_content:
        return "[redacted]"
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "content": _coerce_content_to_text(value.content),
        }
    if isinstance(value, dict):
        return {k: _scrub_value(v, record_content) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_scrub_value(v, record_content) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:  # pragma: no cover
            return str(value)
    return str(value)


def _serialise_tool_result(output: Any, record_content: bool) -> str:
    if isinstance(output, ToolMessage):
        data = {
            "name": getattr(output, "name", None),
            "tool_call_id": _tool_call_id_from_message(output),
            "content": _scrub_value(output.content, record_content),
        }
        return _as_json_attribute(data)
    if isinstance(output, BaseMessage):
        data = {
            "type": output.type,
            "content": _scrub_value(output.content, record_content),
        }
        return _as_json_attribute(data)
    scrubbed = _scrub_value(output, record_content)
    return _as_json_attribute(scrubbed)


def _format_tool_definitions(definitions: Optional[Iterable[Any]]) -> Optional[str]:
    if not definitions:
        return None
    return _as_json_attribute(list(definitions))


def _collect_tool_definitions(*candidates: Any) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, dict):
            iterable: Iterable[Any] = [candidate]
        elif isinstance(candidate, (list, tuple, set)):
            iterable = candidate
        else:
            continue
        for item in iterable:
            if not item or not isinstance(item, dict):
                continue
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            collected.append(item)
    return collected


def _format_documents(
    documents: Optional[Sequence[Document]],
    *,
    record_content: bool,
) -> Optional[str]:
    if not documents:
        return None
    serialised: List[Dict[str, Any]] = []
    for doc in documents:
        entry: Dict[str, Any] = {"metadata": dict(doc.metadata)}
        if record_content:
            entry["content"] = doc.page_content
        serialised.append(entry)
    return _as_json_attribute(serialised)


def _first_non_empty(*values: Any) -> Optional[Any]:
    for value in values:
        if value:
            return value
    return None


def _candidate_from_serialized_id(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if item:
                return str(item)
    if value is not None:
        return str(value)
    return None


def _resolve_agent_name(
    *,
    serialized: Optional[dict[str, Any]],
    metadata: dict[str, Any],
    callback_kwargs: dict[str, Any],
    default: str,
) -> str:
    serialized = serialized or {}
    candidate = _first_non_empty(
        metadata.get("agent_name"),
        metadata.get("langgraph_node"),
        metadata.get("agent_type"),
        callback_kwargs.get("name"),
    )
    resolved = str(candidate) if candidate else None

    generic_markers = {"", _LANGGRAPH_GENERIC_NAME, default}
    if resolved is None or resolved.strip() in generic_markers:
        path = metadata.get("langgraph_path")
        if isinstance(path, (list, tuple)) and path:
            resolved = str(path[-1])

    if resolved is None or resolved.strip() in generic_markers:
        candidate_from_serialized = _candidate_from_serialized_id(
            serialized.get("id")
        ) or _candidate_from_serialized_id(serialized.get("name"))
        if candidate_from_serialized:
            resolved = candidate_from_serialized

    if resolved is None or resolved.strip() == "":
        resolved = default
    return resolved


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_tokens(
    token_usage: Any,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (input_tokens, output_tokens, total_tokens) from usage payloads."""

    def _lookup(keys: Sequence[str]) -> Optional[int]:
        for key in keys:
            if isinstance(token_usage, dict) and key in token_usage:
                return _coerce_int(token_usage[key])
            attr = getattr(token_usage, key, None)
            if attr is not None:
                return _coerce_int(attr)
        return None

    return (
        _lookup(
            (
                "prompt_tokens",
                "input_tokens",
                "inputTokens",
                "inputTokenCount",
                "promptTokenCount",
            )
        ),
        _lookup(
            (
                "completion_tokens",
                "output_tokens",
                "outputTokens",
                "outputTokenCount",
                "completionTokenCount",
            )
        ),
        _lookup(("total_tokens", "totalTokens", "totalTokenCount")),
    )


def _coerce_token_value(value: Any) -> Optional[int]:
    if isinstance(value, (list, tuple, set)):
        total = 0
        found = False
        for item in value:
            coerced = _coerce_token_value(item)
            if coerced is not None:
                total += coerced
                found = True
        return total if found else None
    if isinstance(value, Mapping):
        for key in (
            "value",
            "values",
            "count",
            "token_count",
            "tokenCount",
            "tokens",
            "total",
        ):
            if key in value:
                coerced = _coerce_token_value(value[key])
                if coerced is not None:
                    return coerced
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        total = 0
        found = False
        for item in value:
            coerced = _coerce_token_value(item)
            if coerced is not None:
                total += coerced
                found = True
        return total if found else None
    return _coerce_int(value)


def _normalize_bedrock_usage_dict(
    usage: Any,
) -> Optional[dict[str, int]]:
    if not isinstance(usage, Mapping):
        return None
    normalized: dict[str, int] = {}
    key_variants = {
        "prompt_tokens": (
            "prompt_tokens",
            "input_tokens",
            "inputTokens",
            "inputTokenCount",
            "promptTokenCount",
        ),
        "completion_tokens": (
            "completion_tokens",
            "output_tokens",
            "outputTokens",
            "outputTokenCount",
            "completionTokenCount",
        ),
        "total_tokens": ("total_tokens", "totalTokens", "totalTokenCount"),
    }
    for target_key, variants in key_variants.items():
        for variant in variants:
            if variant in usage:
                value = _coerce_token_value(usage[variant])
                if value is not None:
                    normalized[target_key] = value
                    break
    if not normalized:
        return None
    if "total_tokens" not in normalized:
        input_tokens = normalized.get("prompt_tokens")
        output_tokens = normalized.get("completion_tokens")
        if input_tokens is not None or output_tokens is not None:
            normalized["total_tokens"] = (input_tokens or 0) + (output_tokens or 0)
    return normalized


def _normalize_bedrock_metrics(metrics: Any) -> Optional[dict[str, int]]:
    if not isinstance(metrics, Mapping):
        return None
    normalized: dict[str, int] = {}
    input_tokens = _coerce_token_value(metrics.get("inputTokenCount"))
    output_tokens = _coerce_token_value(metrics.get("outputTokenCount"))
    total_tokens = _coerce_token_value(metrics.get("totalTokenCount"))
    if input_tokens is not None:
        normalized["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        normalized["completion_tokens"] = output_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    elif normalized:
        normalized["total_tokens"] = (normalized.get("prompt_tokens") or 0) + (
            normalized.get("completion_tokens") or 0
        )
    return normalized or None


def _usage_metadata_to_mapping(usage_metadata: Any) -> Optional[Mapping[str, Any]]:
    if usage_metadata is None:
        return None
    if isinstance(usage_metadata, Mapping):
        return usage_metadata
    dict_method = getattr(usage_metadata, "dict", None)
    if callable(dict_method):
        try:
            return dict_method(exclude_none=True)
        except TypeError:
            return dict_method()
    extracted: Dict[str, Any] = {}
    for attr in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        value = getattr(usage_metadata, attr, None)
        if value is not None:
            extracted[attr] = value
    return extracted or None


def _collect_usage_from_generations(
    generations: Sequence[ChatGeneration],
) -> Optional[dict[str, int]]:
    for gen in generations:
        message = getattr(gen, "message", None)
        usage_metadata = getattr(message, "usage_metadata", None)
        mapping = _usage_metadata_to_mapping(usage_metadata)
        if mapping:
            normalized = _normalize_bedrock_usage_dict(mapping)
            if normalized:
                return normalized
        generation_info = getattr(gen, "generation_info", None)
        if isinstance(generation_info, Mapping):
            normalized = _normalize_bedrock_usage_dict(generation_info.get("usage"))
            if normalized:
                return normalized
            normalized = _normalize_bedrock_metrics(
                generation_info.get("amazon-bedrock-invocationMetrics")
            )
            if normalized:
                return normalized
    return None


def _extract_bedrock_usage(
    llm_output: Mapping[str, Any],
    generations: Sequence[ChatGeneration],
) -> Optional[dict[str, int]]:
    usage_candidates: List[Mapping[str, Any]] = []
    if isinstance(llm_output.get("usage"), Mapping):
        usage_candidates.append(llm_output["usage"])
    metrics = _normalize_bedrock_metrics(
        llm_output.get("amazon-bedrock-invocationMetrics")
    )
    if metrics:
        return metrics
    for container_key in (
        "response",
        "response_metadata",
        "additional_kwargs",
        "raw_response",
        "amazon_bedrock",
        "amazon-bedrock",
        "amazonBedrock",
    ):
        container = llm_output.get(container_key)
        if not isinstance(container, Mapping):
            continue
        nested_metrics = _normalize_bedrock_metrics(
            container.get("amazon-bedrock-invocationMetrics")
        )
        if nested_metrics:
            return nested_metrics
        nested_usage = container.get("usage")
        if isinstance(nested_usage, Mapping):
            usage_candidates.append(nested_usage)
    for candidate in usage_candidates:
        normalized = _normalize_bedrock_usage_dict(candidate)
        if normalized:
            return normalized
    return _collect_usage_from_generations(generations)


def _resolve_usage_from_llm_output(
    llm_output: Mapping[str, Any],
    generations: Sequence[ChatGeneration],
) -> tuple[
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[dict[str, int]],
    bool,
]:
    """Return resolved token counts and optionally a normalized payload.

    The final boolean indicates whether the caller should attach the normalized
    payload back onto ``llm_output`` under ``token_usage``.
    """
    candidates: List[tuple[str, Any]] = []
    bedrock_usage = _extract_bedrock_usage(llm_output, generations)
    if bedrock_usage:
        candidates.append(("bedrock", bedrock_usage))
    if llm_output.get("token_usage"):
        candidates.append(("token_usage", llm_output["token_usage"]))
    if isinstance(llm_output.get("usage"), Mapping):
        candidates.append(("usage", llm_output["usage"]))

    for source, payload in candidates:
        (
            input_tokens,
            output_tokens,
            total_tokens,
        ) = _extract_usage_tokens(payload)
        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue
        normalized: dict[str, int] = {}
        if input_tokens is not None:
            normalized["prompt_tokens"] = input_tokens
        if output_tokens is not None:
            normalized["completion_tokens"] = output_tokens
        if total_tokens is None and normalized:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        if total_tokens is not None:
            normalized["total_tokens"] = total_tokens
        should_attach = source != "token_usage"
        return (
            input_tokens,
            output_tokens,
            total_tokens,
            normalized or None,
            should_attach,
        )

    return None, None, None, None, False


def _infer_provider_name(
    serialized: Optional[dict[str, Any]],
    metadata: Optional[dict[str, Any]],
    invocation_params: Optional[dict[str, Any]],
) -> Optional[str]:
    def _contains_bedrock(value: Any) -> bool:
        if isinstance(value, str):
            return "bedrock" in value.lower()
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return any(_contains_bedrock(item) for item in value)
        return False

    provider = (metadata or {}).get("ls_provider")
    if provider:
        provider = provider.lower()
        if provider in {"azure", "azure_openai", "azure-openai"}:
            return "azure.ai.openai"
        if provider in {"openai"}:
            return "openai"
        if provider in {"github"}:
            return "azure.ai.openai"
        if "bedrock" in provider or provider in {"amazon_bedrock", "aws_bedrock"}:
            return "aws.bedrock"
    if invocation_params:
        base_url = invocation_params.get("base_url")
        if isinstance(base_url, str):
            lowered = base_url.lower()
            if "azure" in lowered:
                return "azure.ai.openai"
            if "openai" in lowered:
                return "openai"
            if "ollama" in lowered:
                return "ollama"
            if "bedrock" in lowered or "amazonaws.com" in lowered:
                return "aws.bedrock"
        for key in ("endpoint_url", "service_url"):
            url = invocation_params.get(key)
            if isinstance(url, str) and "bedrock" in url.lower():
                return "aws.bedrock"
        provider_hint = invocation_params.get("provider") or invocation_params.get(
            "provider_name"
        )
        if isinstance(provider_hint, str) and "bedrock" in provider_hint.lower():
            return "aws.bedrock"
    if serialized:
        kwargs = serialized.get("kwargs", {})
        if kwargs.get("azure_endpoint") or kwargs.get("openai_api_base", "").endswith(
            ".azure.com"
        ):
            return "azure.ai.openai"
        identifier_candidates = [
            serialized.get("id"),
            serialized.get("name"),
            kwargs.get("_type"),
            kwargs.get("provider"),
        ]
        if any(_contains_bedrock(candidate) for candidate in identifier_candidates):
            return "aws.bedrock"
    return None


def _infer_server_address(
    serialized: Optional[dict[str, Any]],
    invocation_params: Optional[dict[str, Any]],
) -> Optional[str]:
    base_url = None
    if invocation_params:
        base_url = _first_non_empty(
            invocation_params.get("base_url"),
            invocation_params.get("openai_api_base"),
        )
    if not base_url and serialized:
        kwargs = serialized.get("kwargs", {})
        base_url = _first_non_empty(
            kwargs.get("openai_api_base"),
            kwargs.get("azure_endpoint"),
        )
    if not base_url:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        return parsed.hostname or None
    except Exception:  # pragma: no cover
        return None


def _infer_server_port(
    serialized: Optional[dict[str, Any]],
    invocation_params: Optional[dict[str, Any]],
) -> Optional[int]:
    base_url = None
    if invocation_params:
        base_url = _first_non_empty(
            invocation_params.get("base_url"),
            invocation_params.get("openai_api_base"),
        )
    if not base_url and serialized:
        kwargs = serialized.get("kwargs", {})
        base_url = _first_non_empty(
            kwargs.get("openai_api_base"),
            kwargs.get("azure_endpoint"),
        )
    if not base_url:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        if parsed.port:
            return parsed.port
    except Exception:  # pragma: no cover
        return None
    return None


def _resolve_connection_from_project(
    project_endpoint: Optional[str],
    credential: Optional[Any],
) -> Optional[str]:
    """Resolve Application Insights connection string from an Azure AI project."""
    if not project_endpoint:
        return None
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        LOGGER.warning(
            "azure-identity is required to resolve project endpoints. "
            "Install it or provide APPLICATION_INSIGHTS_CONNECTION_STRING."
        )
        return None
    resolved_credential = credential or DefaultAzureCredential()
    try:
        connection_string, _ = get_service_endpoint_from_project(
            project_endpoint, resolved_credential, "telemetry"
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning(
            "Failed to resolve Application Insights connection string from project "
            "endpoint %s: %s",
            project_endpoint,
            exc,
        )
        return None
    if not connection_string:
        LOGGER.warning(
            "Project %s does not expose a telemetry connection string. "
            "Ensure tracing is enabled for the project.",
            project_endpoint,
        )
        return None
    return connection_string


def _tool_type_from_definition(defn: dict[str, Any]) -> Optional[str]:
    if not defn:
        return None
    if defn.get("type"):
        return str(defn["type"]).lower()
    function = defn.get("function")
    if isinstance(function, dict):
        return function.get("type") or "function"
    return None


@dataclass
class _SpanRecord:
    run_id: str
    span: Span
    operation: str
    parent_run_id: Optional[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    stash: Dict[str, Any] = field(default_factory=dict)


class AzureAIOpenTelemetryTracer(BaseCallbackHandler):
    """LangChain callback handler that emits OpenTelemetry GenAI spans.

    This tracer supports two usage patterns:

    1. **Instance-based** (explicit callbacks):
       ```python
       tracer = AzureAIOpenTelemetryTracer(connection_string="...")
       chain.invoke(input, config={"callbacks": [tracer]})
       ```

    2. **Static autolog** (zero-friction):
       ```python
       AzureAIOpenTelemetryTracer.set_app_insights("InstrumentationKey=...")
       AzureAIOpenTelemetryTracer.set_config({"provider_name": "azure.ai.openai"})
       AzureAIOpenTelemetryTracer.autolog()
       # All LangChain/LangGraph executions now emit spans automatically
       ```

    The static API provides MLflow-style autolog() convenience while maintaining
    full GenAI semantic convention compliance.
    """

    # -------------------------------------------------------------------------
    # Class-level state for static autolog() API
    # -------------------------------------------------------------------------
    _GLOBAL_TRACER_INSTANCE: Optional["AzureAIOpenTelemetryTracer"] = None
    _ACTIVE: bool = False
    _GLOBAL_CONFIG: Dict[str, Any] = {}
    _APP_INSIGHTS_CS: Optional[str] = None
    _STATIC_LOCK: Lock = Lock()

    # -------------------------------------------------------------------------
    # Existing class-level state
    # -------------------------------------------------------------------------
    _azure_monitor_configured: bool = False
    _configure_lock: Lock = Lock()
    _schema_url: str = Schemas.V1_28_0.value

    def __init__(
        self,
        *,
        connection_string: Optional[str] = None,
        enable_content_recording: bool = True,
        project_endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        name: str = "AzureAIOpenTelemetryTracer",
        agent_id: Optional[str] = None,
        provider_name: Optional[str] = None,
        message_keys: Sequence[str] = ("messages",),
        message_paths: Sequence[str] = (),
        trace_all_langgraph_nodes: bool = False,
        ignore_start_node: bool = True,
        compat_create_agent_filtering: bool = True,
        enable_performance_counters: Optional[bool] = None,
    ) -> None:
        """Initialize tracer state and configure Azure Monitor if needed."""
        super().__init__()
        self._name = name
        self._default_agent_id = agent_id
        self._default_provider_name = provider_name
        self._content_recording = enable_content_recording
        self._tracer = otel_trace.get_tracer(name, schema_url=self._schema_url)
        self._message_keys = tuple(message_keys)
        self._message_paths = tuple(message_paths)
        self._trace_all_langgraph_nodes = trace_all_langgraph_nodes
        self._ignore_start_node = ignore_start_node
        self._compat_create_agent_filtering = compat_create_agent_filtering

        if connection_string is None:
            connection_string = _resolve_connection_from_project(
                project_endpoint, credential
            )
        if connection_string is None:
            connection_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")

        if connection_string:
            monitor_kwargs: dict[str, Any] = {"connection_string": connection_string}
            if enable_performance_counters is not None:
                monitor_kwargs["enable_performance_counters"] = enable_performance_counters
            self._configure_azure_monitor(**monitor_kwargs)

        self._spans: Dict[str, _SpanRecord] = {}
        self._lock = Lock()
        self._ignored_runs: set[str] = set()
        self._run_parent_override: Dict[str, Optional[str]] = {}
        self._langgraph_root_by_thread: Dict[str, str] = {}
        self._agent_stack_by_thread: Dict[str, List[str]] = {}

    @contextmanager
    def use_propagated_context(
        self,
        *,
        headers: Mapping[str, str] | None,
    ) -> Iterator[None]:
        """
        Temporarily adopt an upstream trace context extracted from headers.

        This enables scenarios where an HTTP ingress or orchestrator wants to
        ensure the LangGraph spans are correlated with the inbound trace.
        """

        if not headers:
            yield
            return
        try:
            ctx = extract(headers)
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug(
                "Failed to extract OpenTelemetry context from headers; "
                "continuing without propagation.",
                exc_info=True,
            )
            yield
            return
        token = attach(ctx)
        try:
            yield
        finally:
            detach(token)

    def _should_ignore_agent_span(
        self,
        agent_name: Optional[str],
        parent_run_id: Optional[UUID],
        metadata: Optional[dict[str, Any]],
        callback_kwargs: Optional[dict[str, Any]],
    ) -> bool:
        metadata = metadata or {}
        node_name = metadata.get("langgraph_node")
        otel_trace_flag = metadata.get("otel_trace")
        if otel_trace_flag is not None:
            return not bool(otel_trace_flag)
        if self._trace_all_langgraph_nodes:
            if self._ignore_start_node and node_name == _LANGGRAPH_START_NODE:
                return True
            if agent_name and _LANGGRAPH_MIDDLEWARE_PREFIX in agent_name:
                return True
            if metadata.get("otel_agent_span") is False:
                return True
            return False
        if node_name == _LANGGRAPH_START_NODE:
            return True
        otel_flag = metadata.get("otel_agent_span")
        if otel_flag is not None:
            if otel_flag:
                return False
            return True
        if agent_name and _LANGGRAPH_MIDDLEWARE_PREFIX in agent_name:
            return True
        if not self._compat_create_agent_filtering:
            return False
        if parent_run_id is None:
            return False
        if metadata.get("agent_name"):
            return False
        if metadata.get("agent_type"):
            return False
        if agent_name == _LANGGRAPH_GENERIC_NAME:
            return False
        callback_name = str((callback_kwargs or {}).get("name") or "")
        node_label = str(node_name or "")
        if callback_name and node_label and callback_name == node_label:
            return True
        if (
            callback_name == "should_continue"
            and node_label
            and node_label != "coordinator"
        ):
            return True
        if callback_name == _LANGGRAPH_GENERIC_NAME and not metadata.get(
            "otel_agent_span"
        ):
            return True
        return False

    def _resolve_parent_id(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        if parent_run_id is None:
            return None
        candidate: Optional[str] = str(parent_run_id)
        visited: set[str] = set()
        while candidate is not None:
            if candidate in visited:
                return None
            visited.add(candidate)
            if candidate in self._ignored_runs:
                candidate = self._run_parent_override.get(candidate)
                continue
            override = self._run_parent_override.get(candidate)
            if override:
                candidate = override
                continue
            return candidate
        return None

    def _update_parent_attribute(
        self,
        parent_key: Optional[str],
        attr: str,
        value: Any,
    ) -> None:
        if not parent_key or value is None:
            return
        parent_record = self._spans.get(parent_key)
        if not parent_record:
            return
        parent_record.span.set_attribute(attr, value)
        parent_record.attributes[attr] = value

    def _accumulate_usage_to_agent_spans(
        self,
        record: _SpanRecord,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> None:
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return

        thread_key = record.stash.get("thread_id")
        parent_key = record.parent_run_id or self._run_parent_override.get(
            record.run_id
        )
        if not parent_key and thread_key:
            potential_parent = self._langgraph_root_by_thread.get(str(thread_key))
            if potential_parent and potential_parent != record.run_id:
                parent_key = potential_parent
        visited: set[str] = set()
        while parent_key:
            if parent_key in visited:
                break
            visited.add(parent_key)
            parent_record = self._spans.get(parent_key)
            if not parent_record:
                break

            if parent_record.operation == "invoke_agent":
                existing_input = _coerce_int(
                    parent_record.attributes.get(Attrs.USAGE_INPUT_TOKENS)
                )
                existing_output = _coerce_int(
                    parent_record.attributes.get(Attrs.USAGE_OUTPUT_TOKENS)
                )
                existing_total = _coerce_int(
                    parent_record.attributes.get(Attrs.USAGE_TOTAL_TOKENS)
                )

                updated_input: Optional[int] = existing_input
                delta_input: Optional[int] = None
                if input_tokens is not None:
                    updated_input = (existing_input or 0) + input_tokens
                    parent_record.attributes[Attrs.USAGE_INPUT_TOKENS] = updated_input
                    parent_record.span.set_attribute(
                        Attrs.USAGE_INPUT_TOKENS, updated_input
                    )
                    delta_input = input_tokens

                updated_output: Optional[int] = existing_output
                delta_output: Optional[int] = None
                if output_tokens is not None:
                    updated_output = (existing_output or 0) + output_tokens
                    parent_record.attributes[Attrs.USAGE_OUTPUT_TOKENS] = updated_output
                    parent_record.span.set_attribute(
                        Attrs.USAGE_OUTPUT_TOKENS, updated_output
                    )
                    delta_output = output_tokens

                updated_total: Optional[int]
                delta_total: Optional[int] = None
                if total_tokens is not None:
                    updated_total = (existing_total or 0) + total_tokens
                    delta_total = total_tokens
                else:
                    if updated_input is None and updated_output is None:
                        updated_total = existing_total
                    else:
                        inferred_total = (updated_input or 0) + (updated_output or 0)
                        if (
                            existing_total is not None
                            and inferred_total == existing_total
                        ):
                            updated_total = existing_total
                        else:
                            updated_total = inferred_total
                            delta_total = (
                                inferred_total - (existing_total or 0)
                                if existing_total is not None
                                else inferred_total
                            )

                if updated_total is not None:
                    parent_record.attributes[Attrs.USAGE_TOTAL_TOKENS] = updated_total
                    parent_record.span.set_attribute(
                        Attrs.USAGE_TOTAL_TOKENS, updated_total
                    )

                propagated_usage = parent_record.stash.setdefault(
                    "child_usage_propagated",
                    {"input": 0, "output": 0, "total": 0},
                )
                if delta_input:
                    propagated_usage["input"] = (
                        _coerce_int(propagated_usage.get("input")) or 0
                    ) + delta_input
                if delta_output:
                    propagated_usage["output"] = (
                        _coerce_int(propagated_usage.get("output")) or 0
                    ) + delta_output
                if delta_total:
                    propagated_usage["total"] = (
                        _coerce_int(propagated_usage.get("total")) or 0
                    ) + delta_total

            parent_key = parent_record.parent_run_id or self._run_parent_override.get(
                parent_record.run_id
            )
            if not parent_key and thread_key:
                potential_parent = self._langgraph_root_by_thread.get(str(thread_key))
                if potential_parent and potential_parent not in visited:
                    parent_key = potential_parent

    def _propagate_agent_usage_totals(self, record: _SpanRecord) -> None:
        if record.operation != "invoke_agent":
            return
        parent_key = record.parent_run_id or self._run_parent_override.get(
            record.run_id
        )
        if not parent_key:
            thread_key = record.stash.get("thread_id")
            if thread_key:
                potential_parent = self._langgraph_root_by_thread.get(str(thread_key))
                if potential_parent and potential_parent != record.run_id:
                    parent_key = potential_parent
        if not parent_key:
            return
        total_input = _coerce_int(record.attributes.get(Attrs.USAGE_INPUT_TOKENS))
        total_output = _coerce_int(record.attributes.get(Attrs.USAGE_OUTPUT_TOKENS))
        total_tokens = _coerce_int(record.attributes.get(Attrs.USAGE_TOTAL_TOKENS))

        propagated = record.stash.get("child_usage_propagated") or {}
        propagated_input = _coerce_int(propagated.get("input")) or 0
        propagated_output = _coerce_int(propagated.get("output")) or 0
        propagated_total = _coerce_int(propagated.get("total")) or 0

        delta_input: Optional[int] = None
        if total_input is not None:
            remaining_input = total_input - propagated_input
            if remaining_input > 0:
                delta_input = remaining_input

        delta_output: Optional[int] = None
        if total_output is not None:
            remaining_output = total_output - propagated_output
            if remaining_output > 0:
                delta_output = remaining_output

        delta_total: Optional[int] = None
        if total_tokens is not None:
            remaining_total = total_tokens - propagated_total
            if remaining_total > 0:
                delta_total = remaining_total
        else:
            if total_input is not None or total_output is not None:
                inferred_total = (total_input or 0) + (total_output or 0)
                remaining_total = inferred_total - propagated_total
                if remaining_total > 0:
                    delta_total = remaining_total

        if delta_input is None and delta_output is None and delta_total is None:
            return

        self._accumulate_usage_to_agent_spans(
            record, delta_input, delta_output, delta_total
        )

        record.stash["child_usage_propagated"] = {
            "input": total_input if total_input is not None else propagated_input,
            "output": total_output if total_output is not None else propagated_output,
            "total": (
                total_tokens
                if total_tokens is not None
                else (
                    (total_input or 0) + (total_output or 0)
                    if total_input is not None or total_output is not None
                    else propagated_total
                )
            ),
        }

    # ---------------------------------------------------------------------
    # BaseCallbackHandler overrides
    # ---------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle start of a chain/agent invocation."""
        metadata = metadata or {}
        agent_hint = _first_non_empty(
            metadata.get("agent_name"),
            metadata.get("langgraph_node"),
            metadata.get("agent_type"),
            kwargs.get("name"),
        )
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id else None
        if self._should_ignore_agent_span(agent_hint, parent_run_id, metadata, kwargs):
            self._ignored_runs.add(run_key)
            self._run_parent_override[run_key] = parent_key
            return
        attributes: Dict[str, Any] = {
            Attrs.OPERATION_NAME: "invoke_agent",
        }
        resolved_agent_name = _resolve_agent_name(
            serialized=serialized,
            metadata=metadata,
            callback_kwargs=kwargs,
            default=self._name,
        )
        attributes[Attrs.AGENT_NAME] = resolved_agent_name
        node_label = metadata.get("langgraph_node")
        if node_label:
            attributes["metadata.langgraph_node"] = str(node_label)
        agent_id = metadata.get("agent_id")
        if agent_id is not None:
            attributes[Attrs.AGENT_ID] = str(agent_id)
        elif self._default_agent_id:
            attributes[Attrs.AGENT_ID] = self._default_agent_id
        agent_description = metadata.get("agent_description")
        if agent_description:
            attributes[Attrs.AGENT_DESCRIPTION] = str(agent_description)
        thread_identifier = _first_non_empty(
            metadata.get("thread_id"),
            metadata.get("session_id"),
            metadata.get("conversation_id"),
        )
        thread_key = str(thread_identifier) if thread_identifier else None
        if thread_key:
            attributes[Attrs.CONVERSATION_ID] = thread_key
        path = metadata.get("langgraph_path")
        if path:
            attributes["metadata.langgraph_path"] = _as_json_attribute(path)
        for key in (
            Attrs.PROVIDER_NAME,
            Attrs.SERVER_ADDRESS,
            Attrs.SERVER_PORT,
            "service.name",
        ):
            value = metadata.get(key)
            if value is not None:
                attributes[key] = value
        for meta_key, meta_value in metadata.items():
            if meta_key.startswith("gen_ai."):
                attributes[meta_key] = meta_value
        if Attrs.PROVIDER_NAME not in attributes and self._default_provider_name:
            attributes[Attrs.PROVIDER_NAME] = self._default_provider_name

        keys, paths = _effective_message_keys_paths(
            metadata, self._message_keys, self._message_paths
        )
        messages_payload, _ = _extract_messages_payload(
            inputs, message_keys=keys, message_paths=paths
        )
        formatted_messages, system_instr = _prepare_messages(
            messages_payload,
            record_content=self._content_recording,
            include_roles={"user", "assistant", "tool"},
        )
        if formatted_messages:
            attributes[Attrs.INPUT_MESSAGES] = formatted_messages
        if system_instr:
            attributes[Attrs.SYSTEM_INSTRUCTIONS] = system_instr

        is_agent_span = bool(metadata.get("otel_agent_span"))
        effective_parent_run_id = parent_run_id
        resolved_parent = self._resolve_parent_id(parent_run_id)
        original_resolved_parent = resolved_parent
        parent_record = self._spans.get(resolved_parent) if resolved_parent else None
        if (
            is_agent_span
            and parent_record
            and parent_record.operation == "invoke_agent"
        ):
            parent_agent_name = parent_record.attributes.get(Attrs.AGENT_NAME)
            if parent_agent_name != attributes.get(Attrs.AGENT_NAME):
                parent_override_key = parent_record.parent_run_id
                if parent_override_key:
                    try:
                        effective_parent_run_id = UUID(parent_override_key)
                    except (ValueError, TypeError):
                        effective_parent_run_id = None
                else:
                    effective_parent_run_id = None
                resolved_parent = self._resolve_parent_id(effective_parent_run_id)
                parent_record = (
                    self._spans.get(resolved_parent) if resolved_parent else None
                )

        span_name = f"invoke_agent {attributes[Attrs.AGENT_NAME]}"
        self._start_span(
            run_id,
            span_name,
            operation="invoke_agent",
            kind=SpanKind.CLIENT,
            parent_run_id=effective_parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )
        new_record = self._spans.get(run_key)
        allowed_sources = metadata.get("otel_agent_span_allowed")
        if new_record and allowed_sources is not None:
            if isinstance(allowed_sources, str):
                new_record.stash["allowed_agent_sources"] = {allowed_sources}
            else:
                try:
                    new_record.stash["allowed_agent_sources"] = set(allowed_sources)
                except TypeError:
                    LOGGER.debug(
                        "Ignoring non-iterable otel_agent_span_allowed metadata: %r",
                        allowed_sources,
                    )
        if (
            new_record
            and original_resolved_parent
            and (new_record.parent_run_id != original_resolved_parent)
        ):
            self._run_parent_override[run_key] = original_resolved_parent
        if formatted_messages:
            self._update_parent_attribute(
                resolved_parent, Attrs.INPUT_MESSAGES, formatted_messages
            )
        if system_instr:
            self._update_parent_attribute(
                resolved_parent, Attrs.SYSTEM_INSTRUCTIONS, system_instr
            )
        if thread_key:
            self._update_parent_attribute(
                resolved_parent, Attrs.CONVERSATION_ID, thread_key
            )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle completion of a chain/agent invocation."""
        run_key = str(run_id)
        if run_key in self._ignored_runs:
            self._ignored_runs.remove(run_key)
            self._run_parent_override.pop(run_key, None)
            return
        record = self._spans.get(run_key)
        if not record:
            return
        thread_key = record.stash.get("thread_id")
        try:
            keys = self._message_keys
            paths = self._message_paths
            messages_payload, goto = _extract_messages_payload(
                outputs, message_keys=keys, message_paths=paths
            )
            if goto is not None:
                record.span.set_attribute("metadata.langgraph.goto", goto)
                record.attributes["metadata.langgraph.goto"] = goto
            formatted_messages, _ = _prepare_messages(
                messages_payload,
                record_content=self._content_recording,
                include_roles={"assistant"},
            )
            if formatted_messages:
                if record.operation == "invoke_agent":
                    cleaned = _filter_assistant_output(formatted_messages)
                    if cleaned:
                        record.span.set_attribute(Attrs.OUTPUT_MESSAGES, cleaned)
                else:
                    record.span.set_attribute(Attrs.OUTPUT_MESSAGES, formatted_messages)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to serialise chain outputs: %s", exc, exc_info=True)
        record.span.set_status(Status(status_code=StatusCode.OK))
        self._propagate_agent_usage_totals(record)
        if (
            record.operation == "invoke_agent"
            and thread_key
            and self._langgraph_root_by_thread.get(str(thread_key)) == run_key
        ):
            self._langgraph_root_by_thread.pop(str(thread_key), None)
        self._end_span(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Handle errors raised during chain execution."""
        run_key = str(run_id)
        if run_key in self._ignored_runs:
            self._ignored_runs.remove(run_key)
            self._run_parent_override.pop(run_key, None)
            return
        record = self._spans.get(run_key)
        thread_key = record.stash.get("thread_id") if record else None
        if record:
            self._propagate_agent_usage_totals(record)
        self._end_span(
            run_id,
            status=Status(StatusCode.ERROR, str(error)),
            error=error,
        )
        if (
            record
            and record.operation == "invoke_agent"
            and thread_key
            and self._langgraph_root_by_thread.get(str(thread_key)) == run_key
        ):
            self._langgraph_root_by_thread.pop(str(thread_key), None)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Record chat model start metadata."""
        self._handle_model_start(
            serialized=serialized,
            inputs=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            invocation_kwargs=kwargs,
            is_chat_model=True,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Record LLM start metadata."""
        self._handle_model_start(
            serialized=serialized,
            inputs=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            invocation_kwargs=kwargs,
            is_chat_model=False,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Record LLM response attributes and finish the span."""
        record = self._spans.get(str(run_id))
        if not record:
            return

        generations = cast(
            Sequence[Sequence[ChatGeneration]], response.generations or []
        )
        chat_generations: List[ChatGeneration] = []
        if generations:
            chat_generations = [gen for thread in generations for gen in thread]

        if chat_generations:
            messages = [gen.message for gen in chat_generations if gen.message]
            formatted, _ = _prepare_messages(
                messages,
                record_content=self._content_recording,
                include_roles={"assistant"},
            )
            if formatted:
                record.span.set_attribute(Attrs.OUTPUT_MESSAGES, formatted)
                output_type = "text"
                try:
                    parsed = json.loads(formatted)
                    if any(
                        part.get("type") == "tool_call"
                        for msg in parsed
                        for part in msg.get("parts", [])
                    ):
                        output_type = "json"
                except Exception:  # pragma: no cover
                    LOGGER.debug(
                        "Failed to inspect output message for tool calls", exc_info=True
                    )
                record.span.set_attribute(Attrs.OUTPUT_TYPE, output_type)

        finish_reasons: List[str] = []
        for gen in chat_generations:
            info = getattr(gen, "generation_info", {}) or {}
            if info.get("finish_reason"):
                finish_reasons.append(info["finish_reason"])
        if finish_reasons:
            record.span.set_attribute(
                Attrs.RESPONSE_FINISH_REASONS, _as_json_attribute(finish_reasons)
            )

        llm_output_raw = getattr(response, "llm_output", {}) or {}
        llm_output = llm_output_raw if isinstance(llm_output_raw, Mapping) else {}
        (
            input_tokens,
            output_tokens,
            total_tokens,
            normalized_usage,
            should_attach_usage,
        ) = _resolve_usage_from_llm_output(llm_output, chat_generations)
        if normalized_usage and should_attach_usage:
            try:
                if hasattr(llm_output_raw, "__setitem__"):
                    llm_output_raw["token_usage"] = dict(normalized_usage)
                elif hasattr(llm_output_raw, "__dict__"):
                    setattr(llm_output_raw, "token_usage", dict(normalized_usage))
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug(
                    "Failed to attach normalized usage to llm_output", exc_info=True
                )
        if (
            input_tokens is not None
            or output_tokens is not None
            or total_tokens is not None
        ):
            if input_tokens is not None:
                record.span.set_attribute(Attrs.USAGE_INPUT_TOKENS, input_tokens)
                record.attributes[Attrs.USAGE_INPUT_TOKENS] = input_tokens
            if output_tokens is not None:
                record.span.set_attribute(Attrs.USAGE_OUTPUT_TOKENS, output_tokens)
                record.attributes[Attrs.USAGE_OUTPUT_TOKENS] = output_tokens
            if total_tokens is None and (
                input_tokens is not None or output_tokens is not None
            ):
                total_tokens = (input_tokens or 0) + (output_tokens or 0)
            if total_tokens is not None:
                record.span.set_attribute(Attrs.USAGE_TOTAL_TOKENS, total_tokens)
                record.attributes[Attrs.USAGE_TOTAL_TOKENS] = total_tokens
            self._accumulate_usage_to_agent_spans(
                record, input_tokens, output_tokens, total_tokens
            )
        if "id" in llm_output:
            record.span.set_attribute(Attrs.RESPONSE_ID, str(llm_output["id"]))
        if "model_name" in llm_output:
            record.span.set_attribute(Attrs.RESPONSE_MODEL, llm_output["model_name"])
        if llm_output.get("system_fingerprint"):
            record.span.set_attribute(
                Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
                llm_output["system_fingerprint"],
            )
        if llm_output.get("service_tier"):
            record.span.set_attribute(
                Attrs.OPENAI_RESPONSE_SERVICE_TIER, llm_output["service_tier"]
            )

        model_name = llm_output.get("model_name") or record.attributes.get(
            Attrs.REQUEST_MODEL
        )
        if model_name:
            record.span.update_name(f"{record.operation} {model_name}")

        record.span.set_status(Status(StatusCode.OK))
        self._end_span(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Mark the LLM span as errored."""
        self._end_span(
            run_id,
            status=Status(StatusCode.ERROR, str(error)),
            error=error,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a span representing tool execution."""
        metadata = metadata or {}
        resolved_parent = self._resolve_parent_id(parent_run_id)
        thread_identifier = _first_non_empty(
            metadata.get("thread_id"),
            metadata.get("session_id"),
            metadata.get("conversation_id"),
        )
        thread_key = str(thread_identifier) if thread_identifier else None
        tool_name = _first_non_empty(
            serialized.get("name"),
            metadata.get("tool_name"),
            kwargs.get("name"),
        )
        attributes = {
            Attrs.OPERATION_NAME: "execute_tool",
            Attrs.TOOL_NAME: tool_name or "tool",
        }
        if serialized.get("description"):
            attributes[Attrs.TOOL_DESCRIPTION] = serialized["description"]
        tool_type = _tool_type_from_definition(serialized)
        if tool_type:
            attributes[Attrs.TOOL_TYPE] = tool_type
        tool_id = (inputs or {}).get("tool_call_id") or metadata.get("tool_call_id")
        if tool_id:
            attributes[Attrs.TOOL_CALL_ID] = str(tool_id)
        if inputs:
            attributes[Attrs.TOOL_CALL_ARGUMENTS] = _as_json_attribute(inputs)
        elif input_str:
            attributes[Attrs.TOOL_CALL_ARGUMENTS] = input_str
        parent_provider = None
        if resolved_parent and resolved_parent in self._spans:
            parent_provider = self._spans[resolved_parent].attributes.get(
                Attrs.PROVIDER_NAME
            )
        if parent_provider:
            attributes[Attrs.PROVIDER_NAME] = parent_provider
        elif self._default_provider_name:
            attributes[Attrs.PROVIDER_NAME] = self._default_provider_name

        self._start_span(
            run_id,
            name=f"execute_tool {tool_name}" if tool_name else "execute_tool",
            operation="execute_tool",
            kind=SpanKind.INTERNAL,
            parent_run_id=parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Finalize a tool span with results."""
        record = self._spans.get(str(run_id))
        if not record:
            return
        if output is not None:
            record.span.set_attribute(
                Attrs.TOOL_CALL_RESULT,
                _serialise_tool_result(output, self._content_recording),
            )
        record.span.set_status(Status(StatusCode.OK))
        self._end_span(run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Mark a tool span as errored."""
        self._end_span(
            run_id,
            status=Status(StatusCode.ERROR, str(error)),
            error=error,
        )

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Cache tool context emitted from agent actions."""
        # AgentAction is emitted before tool execution; store arguments so that
        # subsequent tool spans can include more context.
        resolved_parent = self._resolve_parent_id(parent_run_id)
        record = self._spans.get(resolved_parent) if resolved_parent else None
        if record is not None:
            record.stash.setdefault("pending_actions", {})[str(run_id)] = {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "log": action.log,
            }
            last_chat = record.stash.get("last_chat_run")
            if last_chat:
                self._run_parent_override[str(run_id)] = last_chat

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Close an agent span and record outputs."""
        record = self._spans.get(str(run_id))
        if not record:
            return
        if finish.return_values:
            record.span.set_attribute(
                Attrs.OUTPUT_MESSAGES, _as_json_attribute(finish.return_values)
            )
        record.span.set_status(Status(StatusCode.OK))
        self._propagate_agent_usage_totals(record)
        self._end_span(run_id)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Start a retriever span."""
        metadata = metadata or {}
        serialized = serialized or {}
        resolved_parent = self._resolve_parent_id(parent_run_id)
        thread_identifier = _first_non_empty(
            metadata.get("thread_id"),
            metadata.get("session_id"),
            metadata.get("conversation_id"),
        )
        thread_key = str(thread_identifier) if thread_identifier else None
        attributes = {
            Attrs.OPERATION_NAME: "execute_tool",
            Attrs.TOOL_NAME: serialized.get("name", "retriever"),
            Attrs.TOOL_DESCRIPTION: serialized.get("description", "retriever"),
            Attrs.TOOL_TYPE: "retriever",
            Attrs.RETRIEVER_QUERY: query,
        }
        parent_provider = None
        if resolved_parent and resolved_parent in self._spans:
            parent_provider = self._spans[resolved_parent].attributes.get(
                Attrs.PROVIDER_NAME
            )
        if parent_provider:
            attributes[Attrs.PROVIDER_NAME] = parent_provider
        self._start_span(
            run_id,
            name=f"execute_tool {serialized.get('name', 'retriever')}",
            operation="execute_tool",
            kind=SpanKind.INTERNAL,
            parent_run_id=parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Record retriever results and close the span."""
        record = self._spans.get(str(run_id))
        if not record:
            return
        formatted = _format_documents(documents, record_content=self._content_recording)
        if formatted:
            record.span.set_attribute(Attrs.RETRIEVER_RESULTS, formatted)
            record.attributes[Attrs.RETRIEVER_RESULTS] = formatted
        record.span.set_status(Status(StatusCode.OK))
        self._end_span(run_id)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Mark a retriever span as errored."""
        self._end_span(
            run_id,
            status=Status(StatusCode.ERROR, str(error)),
            error=error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_model_start(
        self,
        *,
        serialized: dict[str, Any],
        inputs: Any,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        metadata: Optional[dict[str, Any]],
        invocation_kwargs: dict[str, Any],
        is_chat_model: bool,
    ) -> None:
        invocation_params = invocation_kwargs.get("invocation_params") or {}
        metadata = metadata or {}
        thread_identifier = _first_non_empty(
            metadata.get("thread_id"),
            metadata.get("session_id"),
            metadata.get("conversation_id"),
        )
        thread_key = str(thread_identifier) if thread_identifier else None
        model_name = _first_non_empty(
            invocation_params.get("model"),
            invocation_params.get("model_name"),
            (serialized.get("kwargs", {}) or {}).get("model"),
            (serialized.get("kwargs", {}) or {}).get("model_name"),
        )
        provider = _infer_provider_name(serialized, metadata, invocation_params)
        attributes: Dict[str, Any] = {
            Attrs.OPERATION_NAME: "chat" if is_chat_model else "text_completion",
        }
        if provider:
            attributes[Attrs.PROVIDER_NAME] = provider
        elif self._default_provider_name:
            attributes[Attrs.PROVIDER_NAME] = self._default_provider_name
        if model_name:
            attributes[Attrs.REQUEST_MODEL] = model_name

        for attr_name, key in [
            (Attrs.REQUEST_MAX_TOKENS, "max_tokens"),
            (Attrs.REQUEST_MAX_INPUT_TOKENS, "max_input_tokens"),
            (Attrs.REQUEST_MAX_OUTPUT_TOKENS, "max_output_tokens"),
            (Attrs.REQUEST_TEMPERATURE, "temperature"),
            (Attrs.REQUEST_TOP_P, "top_p"),
            (Attrs.REQUEST_TOP_K, "top_k"),
            (Attrs.REQUEST_FREQ_PENALTY, "frequency_penalty"),
            (Attrs.REQUEST_PRES_PENALTY, "presence_penalty"),
            (Attrs.REQUEST_CHOICE_COUNT, "n"),
            (Attrs.REQUEST_SEED, "seed"),
        ]:
            if key in invocation_params and invocation_params[key] is not None:
                attributes[attr_name] = invocation_params[key]

        if invocation_params.get("stop"):
            attributes[Attrs.REQUEST_STOP] = _as_json_attribute(
                invocation_params["stop"]
            )
        if invocation_params.get("response_format"):
            attributes[Attrs.REQUEST_ENCODING_FORMATS] = _as_json_attribute(
                invocation_params["response_format"]
            )

        formatted_input, system_instr = _prepare_messages(
            inputs,
            record_content=self._content_recording,
            include_roles={"user", "assistant", "tool"},
        )
        if formatted_input:
            attributes[Attrs.INPUT_MESSAGES] = formatted_input
        if system_instr:
            attributes[Attrs.SYSTEM_INSTRUCTIONS] = system_instr

        serialized_kwargs = serialized.get("kwargs") or {}
        if not isinstance(serialized_kwargs, dict):
            serialized_kwargs = {}
        tool_definitions = _collect_tool_definitions(
            invocation_params.get("tools"),
            invocation_params.get("functions"),
            invocation_kwargs.get("tools"),
            invocation_kwargs.get("functions"),
            serialized_kwargs.get("tools"),
            serialized_kwargs.get("functions"),
        )
        tool_definitions_json = None
        if tool_definitions:
            tool_definitions_json = _format_tool_definitions(tool_definitions)
            attributes[Attrs.TOOL_DEFINITIONS] = tool_definitions_json

        server_address = _infer_server_address(serialized, invocation_params)
        if server_address:
            attributes[Attrs.SERVER_ADDRESS] = server_address
        server_port = _infer_server_port(serialized, invocation_params)
        if server_port:
            attributes[Attrs.SERVER_PORT] = server_port

        service_tier = invocation_params.get("service_tier")
        if service_tier:
            attributes[Attrs.OPENAI_REQUEST_SERVICE_TIER] = service_tier

        operation_name = attributes[Attrs.OPERATION_NAME]
        span_name = f"{operation_name} {model_name}" if model_name else operation_name
        resolved_parent = self._resolve_parent_id(parent_run_id)
        self._start_span(
            run_id,
            name=span_name,
            operation=attributes[Attrs.OPERATION_NAME],
            kind=SpanKind.CLIENT,
            parent_run_id=parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )
        span_record = self._spans.get(str(run_id))  # noqa: F841 - used for future span mutations
        if provider:
            self._update_parent_attribute(
                resolved_parent, Attrs.PROVIDER_NAME, provider
            )
        if formatted_input:
            self._update_parent_attribute(
                resolved_parent, Attrs.INPUT_MESSAGES, formatted_input
            )
        if system_instr:
            self._update_parent_attribute(
                resolved_parent, Attrs.SYSTEM_INSTRUCTIONS, system_instr
            )
        if tool_definitions_json and resolved_parent:
            self._update_parent_attribute(
                resolved_parent, Attrs.TOOL_DEFINITIONS, tool_definitions_json
            )
        chat_run_key = str(run_id)
        if resolved_parent and resolved_parent in self._spans:
            self._spans[resolved_parent].stash["last_chat_run"] = chat_run_key
            # Store the chat span's context so it can be used as parent for tool calls
            # even after the chat span ends (since chat ends before tools start in LangChain)
            chat_record = self._spans.get(chat_run_key)
            if chat_record:
                self._spans[resolved_parent].stash["last_chat_context"] = set_span_in_context(chat_record.span)

    def _start_span(
        self,
        run_id: UUID,
        name: str,
        *,
        operation: str,
        kind: SpanKind,
        parent_run_id: Optional[UUID],
        attributes: Optional[Dict[str, Any]] = None,
        thread_key: Optional[str] = None,
    ) -> None:
        run_key = str(run_id)
        resolved_parent_key = self._resolve_parent_id(parent_run_id)
        parent_context = None
        parent_record = None
        thread_str = str(thread_key) if thread_key else None
        stack = self._agent_stack_by_thread.get(thread_str) if thread_str else None
        parent_source = "explicit"
        if resolved_parent_key and resolved_parent_key in self._spans:
            parent_record = self._spans[resolved_parent_key]
            actual_parent_record = parent_record
            if (
                operation == "execute_tool"
                and parent_record.operation == "invoke_agent"
            ):
                # For tool execution, prefer the last chat span as parent
                # This creates the hierarchy: invoke_agent -> chat -> execute_tool
                last_chat = parent_record.stash.get("last_chat_run")
                if last_chat and last_chat in self._spans:
                    # Chat span is still active - use it as parent
                    actual_parent_record = self._spans[last_chat]
                    resolved_parent_key = last_chat
                    parent_context = set_span_in_context(actual_parent_record.span)
                elif "last_chat_context" in parent_record.stash:
                    # Chat span has ended but we stored its context - use that
                    parent_context = parent_record.stash["last_chat_context"]
                    resolved_parent_key = last_chat  # Keep the chat run key for record
                else:
                    parent_context = set_span_in_context(actual_parent_record.span)
            else:
                parent_context = set_span_in_context(actual_parent_record.span)
        if resolved_parent_key is None:
            fallback_key = None
            if thread_str:
                if operation == "invoke_agent":
                    if stack:
                        fallback_key = stack[-1]
                    else:
                        fallback_key = self._langgraph_root_by_thread.get(thread_str)
                else:
                    if stack:
                        fallback_key = stack[-1]
            if fallback_key == run_key:
                fallback_key = None
            if fallback_key and fallback_key in self._spans:
                parent_record = self._spans[fallback_key]
                resolved_parent_key = fallback_key
                parent_context = set_span_in_context(parent_record.span)
                parent_source = "stack"
            else:
                current_span = get_current_span()
                if current_span and current_span.get_span_context().is_valid:
                    parent_context = set_span_in_context(current_span)
                    parent_source = "current"
                else:
                    parent_source = "none"
        if (
            operation != "invoke_agent"
            and stack
            and stack[-1] != run_key
            and stack[-1] in self._spans
            and stack[-1] != resolved_parent_key
        ):
            resolved_parent_key = stack[-1]
            parent_record = self._spans[resolved_parent_key]
            parent_context = set_span_in_context(parent_record.span)
            parent_source = "stack_override"

        span = self._tracer.start_span(
            name=name,
            context=parent_context,
            kind=kind,
            attributes=attributes or {},
        )
        span_record = _SpanRecord(
            run_id=run_key,
            span=span,
            operation=operation,
            parent_run_id=resolved_parent_key,
            attributes=attributes or {},
        )
        self._spans[run_key] = span_record
        self._run_parent_override[run_key] = resolved_parent_key
        if thread_key:
            span_record.stash["thread_id"] = thread_key
            if (
                resolved_parent_key is None
                and str(thread_key) not in self._langgraph_root_by_thread
            ):
                self._langgraph_root_by_thread[str(thread_key)] = run_key
            if operation == "invoke_agent":
                stack = self._agent_stack_by_thread.setdefault(str(thread_key), [])
                stack.append(run_key)
        if resolved_parent_key and resolved_parent_key in self._spans:
            conv_id = self._spans[resolved_parent_key].attributes.get(
                Attrs.CONVERSATION_ID
            )
            if conv_id and Attrs.CONVERSATION_ID not in (attributes or {}):
                span.set_attribute(Attrs.CONVERSATION_ID, conv_id)
                self._spans[run_key].attributes[Attrs.CONVERSATION_ID] = conv_id
        if _DEBUG_THREAD_STACK:
            LOGGER.info(
                "start span op=%s run=%s thread=%s parent=%s source=%s stack=%s",
                operation,
                run_key,
                thread_key,
                resolved_parent_key,
                parent_source,
                list(stack or []),
            )

    def _end_span(
        self,
        run_id: UUID,
        *,
        status: Optional[Status] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        record = self._spans.pop(str(run_id), None)
        if not record:
            return
        if error and status is None:
            status = Status(StatusCode.ERROR, str(error))
            record.span.set_attribute(Attrs.ERROR_TYPE, error.__class__.__name__)
        if status:
            record.span.set_status(status)
        thread_key = record.stash.get("thread_id")
        if record.operation == "invoke_agent" and thread_key:
            stack = self._agent_stack_by_thread.get(str(thread_key))
            if stack:
                if stack[-1] == record.run_id:
                    stack.pop()
                if not stack:
                    self._agent_stack_by_thread.pop(str(thread_key), None)
            if _DEBUG_THREAD_STACK:
                LOGGER.info(
                    "end span op=%s run=%s thread=%s stack_after=%s",
                    record.operation,
                    record.run_id,
                    thread_key,
                    list(self._agent_stack_by_thread.get(str(thread_key), []))
                    if thread_key
                    else [],
                )
        record.span.end()
        self._run_parent_override.pop(str(run_id), None)

    @classmethod
    def _configure_azure_monitor(cls, *, connection_string: str, **kwargs: Any) -> None:
        with cls._configure_lock:
            if cls._azure_monitor_configured:
                return
            configure_azure_monitor(connection_string=connection_string, **kwargs)
            cls._azure_monitor_configured = True

    # -------------------------------------------------------------------------
    # Static autolog() API - MLflow-style zero-friction instrumentation
    # -------------------------------------------------------------------------

    @classmethod
    def set_app_insights(cls, connection_string: str) -> None:
        """Configure Azure Application Insights connection string.

        This can be called before or after autolog(). If called after autolog()
        is already active, the Azure Monitor exporter will be configured
        immediately.

        Args:
            connection_string: Azure Application Insights connection string
                (e.g., "InstrumentationKey=...;IngestionEndpoint=...")
        """
        with cls._STATIC_LOCK:
            cls._APP_INSIGHTS_CS = connection_string
            # If autolog is already active, configure Azure Monitor immediately
            if cls._ACTIVE and connection_string:
                try:
                    cls._configure_azure_monitor(connection_string=connection_string)
                except Exception:
                    LOGGER.exception("Failed configuring Azure Monitor after set_app_insights")

    @classmethod
    def set_config(cls, cfg: Dict[str, Any]) -> None:
        """Merge user configuration into the global config store.

        Configuration precedence (highest to lowest):
        1. Per-call overrides passed to autolog()
        2. Values from set_config()
        3. Environment variables
        4. DEFAULT_CONFIG

        Args:
            cfg: Dictionary of configuration options to merge.
                See DEFAULT_CONFIG for available keys.
        """
        with cls._STATIC_LOCK:
            if cfg:
                # Validate keys against known config options
                unknown_keys = set(cfg.keys()) - set(DEFAULT_CONFIG.keys())
                if unknown_keys:
                    LOGGER.warning(
                        "Unknown configuration keys will be ignored: %s",
                        unknown_keys,
                    )
                cls._GLOBAL_CONFIG.update(cfg)

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return the effective merged configuration.

        Returns:
            Dictionary containing all configuration options with precedence applied.
        """
        # Start with defaults, overlay global config
        effective = dict(DEFAULT_CONFIG)
        effective.update(cls._GLOBAL_CONFIG)

        # Check environment variable overrides
        env_mappings = {
            "GENAI_REDACT_MESSAGES": ("redact_messages", lambda v: v.lower() in ("1", "true", "yes")),
            "GENAI_REDACT_TOOL_ARGUMENTS": ("redact_tool_arguments", lambda v: v.lower() in ("1", "true", "yes")),
            "GENAI_REDACT_TOOL_RESULTS": ("redact_tool_results", lambda v: v.lower() in ("1", "true", "yes")),
            "GENAI_ENABLE_CONTENT_RECORDING": ("enable_content_recording", lambda v: v.lower() in ("1", "true", "yes")),
            "GENAI_SAMPLING_RATE": ("sampling_rate", float),
            "GENAI_PROVIDER_NAME": ("provider_name", str),
        }
        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    effective[config_key] = converter(env_value)
                except (ValueError, TypeError):
                    LOGGER.warning(
                        "Invalid value for environment variable %s: %s",
                        env_var,
                        env_value,
                    )

        return effective

    @classmethod
    def autolog(cls, **overrides: Any) -> None:
        """Activate automatic instrumentation for all LangChain/LangGraph executions.

        After calling this method, all LangChain chains, agents, and LangGraph
        graphs will automatically emit OpenTelemetry spans without requiring
        explicit `callbacks=[tracer]` configuration.

        This method is idempotent - calling it multiple times will merge any
        new overrides but not create duplicate tracers.

        Args:
            **overrides: Configuration overrides that take highest precedence.
                See DEFAULT_CONFIG for available options.

        Example:
            >>> AzureAIOpenTelemetryTracer.set_app_insights("InstrumentationKey=...")
            >>> AzureAIOpenTelemetryTracer.autolog(provider_name="azure.ai.openai")
            >>> # All subsequent LangChain calls are now traced
        """
        with cls._STATIC_LOCK:
            if cls._ACTIVE:
                LOGGER.info(
                    "AzureAIOpenTelemetryTracer.autolog() already active; "
                    "merging overrides if provided."
                )
                if overrides:
                    cls._GLOBAL_CONFIG.update(overrides)
                    # Update the existing tracer's content recording setting if changed
                    if cls._GLOBAL_TRACER_INSTANCE:
                        effective = cls.get_config()
                        cls._GLOBAL_TRACER_INSTANCE._content_recording = effective.get(
                            "enable_content_recording", True
                        ) and not effective.get("redact_messages", False)
                return

            # Merge overrides into global config
            if overrides:
                cls._GLOBAL_CONFIG.update(overrides)

            effective = cls.get_config()

            # Build tracer instance with effective config
            instance = cls(
                name=effective.get("tracer_name", "azure_ai_genai_tracer"),
                provider_name=effective.get("provider_name"),
                connection_string=cls._APP_INSIGHTS_CS,
                enable_content_recording=effective.get("enable_content_recording", True)
                and not effective.get("redact_messages", False),
                message_keys=effective.get("message_keys", ("messages",)),
                message_paths=effective.get("message_paths", ()),
                trace_all_langgraph_nodes=effective.get("trace_all_langgraph_nodes", False),
                ignore_start_node=effective.get("ignore_start_node", True),
                compat_create_agent_filtering=effective.get("compat_create_agent_filtering", True),
                enable_performance_counters=effective.get("enable_performance_counters"),
            )

            cls._GLOBAL_TRACER_INSTANCE = instance
            cls._ACTIVE = True

            # Register callback globally with LangChain
            cls._register_global_callback(instance)

            LOGGER.info(
                "AzureAIOpenTelemetryTracer.autolog() activated with config: %s",
                {k: v for k, v in effective.items() if k != "connection_string"},
            )

    @classmethod
    def _register_global_callback(cls, instance: "AzureAIOpenTelemetryTracer") -> None:
        """Register the tracer instance as a global callback handler.

        This uses LangChain's callback manager to automatically include the
        tracer in all chain/agent/graph invocations.
        """
        try:
            from langchain_core.tracers.context import register_configure_hook

            # Newer langchain_core uses ContextVar-based configure hooks.
            # We register our own ContextVar and set it to the tracer instance so
            # CallbackManager.configure() automatically includes it.
            ctx_token = _LANGCHAIN_TRACER_CONTEXT_VAR.set(instance)
            try:
                register_configure_hook(_LANGCHAIN_TRACER_CONTEXT_VAR, inheritable=True)
                cls._GLOBAL_CONFIG["_registered_hook"] = (
                    "context_var",
                    _LANGCHAIN_TRACER_CONTEXT_VAR,
                )
                cls._GLOBAL_CONFIG["_registered_hook_token"] = ctx_token
                LOGGER.debug("Registered tracer via context-var configure hook")
                return
            except TypeError:
                # Fall back to older langchain_core versions where
                # register_configure_hook() accepts a hook function.
                _LANGCHAIN_TRACER_CONTEXT_VAR.reset(ctx_token)

            from langchain_core.callbacks.manager import CallbackManager

            # Register a configure hook that adds our tracer to every run
            def _add_tracer_hook(
                local_callbacks: Any,
                inheritable_callbacks: Any,
                local_tags: Any,
                inheritable_tags: Any,
                local_metadata: Any,
                inheritable_metadata: Any,
            ) -> tuple:
                """Hook that injects our tracer into callback configuration."""
                # Add our tracer to inheritable callbacks
                if inheritable_callbacks is None:
                    inheritable_callbacks = [instance]
                elif isinstance(inheritable_callbacks, list):
                    if instance not in inheritable_callbacks:
                        inheritable_callbacks = inheritable_callbacks + [instance]
                elif isinstance(inheritable_callbacks, CallbackManager):
                    if instance not in inheritable_callbacks.handlers:
                        inheritable_callbacks.add_handler(instance, inherit=True)
                return (
                    local_callbacks,
                    inheritable_callbacks,
                    local_tags,
                    inheritable_tags,
                    local_metadata,
                    inheritable_metadata,
                )

            register_configure_hook(_add_tracer_hook)
            cls._GLOBAL_CONFIG["_registered_hook"] = ("hook_fn", _add_tracer_hook)
            LOGGER.debug("Registered tracer via configure hook")

        except ImportError:
            LOGGER.warning(
                "Could not import LangChain callback manager. "
                "Global callback registration may not work as expected."
            )
        except Exception:
            LOGGER.exception("Failed to register global callback handler")

    @classmethod
    def is_active(cls) -> bool:
        """Check whether autolog() instrumentation is currently active.

        Returns:
            True if autolog() has been called and not yet shut down.
        """
        return cls._ACTIVE

    @classmethod
    def shutdown(cls) -> None:
        """Deactivate instrumentation and flush pending spans.

        This method:
        1. Forces a flush of all pending spans to exporters
        2. Removes the global callback registration
        3. Clears internal state

        It's safe to call autolog() again after shutdown().
        """
        with cls._STATIC_LOCK:
            if not cls._ACTIVE:
                LOGGER.debug("shutdown() called but autolog() was not active")
                return

            # Force flush the tracer provider
            try:
                provider = otel_trace.get_tracer_provider()
                if hasattr(provider, "force_flush"):
                    provider.force_flush(timeout_millis=5000)
            except Exception:
                LOGGER.debug("Tracer provider flush failed", exc_info=True)

            # Clean up the global tracer instance
            if cls._GLOBAL_TRACER_INSTANCE:
                # End any dangling spans
                for run_key in list(cls._GLOBAL_TRACER_INSTANCE._spans.keys()):
                    try:
                        record = cls._GLOBAL_TRACER_INSTANCE._spans.pop(run_key, None)
                        if record:
                            record.span.end()
                    except Exception:
                        LOGGER.debug("Failed to end span %s", run_key, exc_info=True)

            # Unregister the configure hook if we registered one
            registered_hook = cls._GLOBAL_CONFIG.pop("_registered_hook", None)
            if registered_hook:
                try:
                    from langchain_core.tracers.context import (
                        _configure_hooks,
                    )
                    if (
                        isinstance(registered_hook, tuple)
                        and len(registered_hook) == 2
                        and registered_hook[0] == "hook_fn"
                    ):
                        hook_fn = registered_hook[1]
                        if hook_fn in _configure_hooks:
                            _configure_hooks.remove(hook_fn)
                    elif (
                        isinstance(registered_hook, tuple)
                        and len(registered_hook) == 2
                        and registered_hook[0] == "context_var"
                    ):
                        context_var = registered_hook[1]
                        _configure_hooks[:] = [
                            hook
                            for hook in _configure_hooks
                            if not (isinstance(hook, tuple) and hook and hook[0] is context_var)
                        ]
                        ctx_token = cls._GLOBAL_CONFIG.pop(
                            "_registered_hook_token", None
                        )
                        if ctx_token is not None:
                            try:
                                context_var.reset(ctx_token)
                            except Exception:
                                LOGGER.debug(
                                    "Failed to reset tracer context var",
                                    exc_info=True,
                                )
                except Exception:
                    LOGGER.debug("Failed to unregister configure hook", exc_info=True)

            cls._GLOBAL_TRACER_INSTANCE = None
            cls._ACTIVE = False

            LOGGER.info("AzureAIOpenTelemetryTracer.shutdown() completed")

    @classmethod
    def add_tags(cls, tags: Dict[str, Any]) -> None:
        """Add trace-level tags to root agent spans.

        Tags are added to all currently active root spans (spans without a parent).
        This is useful for adding request-specific metadata like user IDs or
        session identifiers.

        Args:
            tags: Dictionary of tag names and values to add.
        """
        if not cls._ACTIVE or not tags:
            return
        inst = cls._GLOBAL_TRACER_INSTANCE
        if not inst:
            return
        inst._add_root_tags(tags)

    def _add_root_tags(self, tags: Dict[str, Any]) -> None:
        """Instance method to add tags to root spans."""
        for record in list(self._spans.values()):
            if record.parent_run_id is None and record.operation == "invoke_agent":
                for key, value in tags.items():
                    try:
                        record.span.set_attribute(key, value)
                        record.attributes[key] = value
                    except Exception:
                        LOGGER.debug(
                            "Failed to set tag %s on span", key, exc_info=True
                        )

    @classmethod
    def update_redaction_rules(
        cls,
        *,
        redact_messages: Optional[bool] = None,
        redact_tool_arguments: Optional[bool] = None,
        redact_tool_results: Optional[bool] = None,
    ) -> None:
        """Update redaction settings at runtime.

        Changes take effect immediately for subsequent spans. Existing spans
        are not modified.

        Args:
            redact_messages: If True, redact message content in spans.
            redact_tool_arguments: If True, redact tool call arguments.
            redact_tool_results: If True, redact tool call results.
        """
        with cls._STATIC_LOCK:
            if redact_messages is not None:
                cls._GLOBAL_CONFIG["redact_messages"] = redact_messages
            if redact_tool_arguments is not None:
                cls._GLOBAL_CONFIG["redact_tool_arguments"] = redact_tool_arguments
            if redact_tool_results is not None:
                cls._GLOBAL_CONFIG["redact_tool_results"] = redact_tool_results

            # Update the active tracer instance if present
            if cls._GLOBAL_TRACER_INSTANCE and redact_messages is not None:
                effective = cls.get_config()
                cls._GLOBAL_TRACER_INSTANCE._content_recording = (
                    effective.get("enable_content_recording", True)
                    and not effective.get("redact_messages", False)
                )

    @classmethod
    def force_flush(cls, timeout_millis: int = 5000) -> bool:
        """Force flush all pending spans to exporters.

        This is useful after batch workflows to ensure all telemetry is
        exported before the process exits.

        Args:
            timeout_millis: Maximum time to wait for flush completion.

        Returns:
            True if flush completed successfully, False otherwise.
        """
        try:
            provider = otel_trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                return provider.force_flush(timeout_millis=timeout_millis)
            return True
        except Exception:
            LOGGER.debug("force_flush() failed", exc_info=True)
            return False

    @classmethod
    def get_tracer_instance(cls) -> Optional["AzureAIOpenTelemetryTracer"]:
        """Get the global tracer instance if autolog() is active.

        Returns:
            The active tracer instance, or None if autolog() has not been called.
        """
        return cls._GLOBAL_TRACER_INSTANCE if cls._ACTIVE else None
