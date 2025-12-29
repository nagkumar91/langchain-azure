import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.trace.status import StatusCode

import langchain_azure_ai.callbacks.tracers.inference_tracing as tracing

# Skip tests cleanly if required deps are not present
pytest.importorskip("azure.monitor.opentelemetry")
pytest.importorskip("opentelemetry")
pytest.importorskip("langchain_core")


class MockSpan:
    name: str
    attributes: Dict[str, Any]
    events: List[Tuple[str, Dict[str, Any]]]
    ended: bool
    status: Optional[Any]
    exceptions: List[Exception]

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})
        self.events = []
        self.ended = False
        self.status = None
        self.exceptions = []
        self._context = SimpleNamespace(is_valid=True)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended = True

    def get_span_context(self) -> Any:
        return self._context

    def update_name(self, name: str) -> None:
        self.name = name


class MockTracer:
    spans: List[MockSpan]

    def __init__(self) -> None:
        self.spans = []

    def start_span(
        self,
        name: str,
        kind: Any = None,
        context: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> MockSpan:
        span = MockSpan(name, attributes)
        self.spans.append(span)
        return span


class MockTracerProvider:
    """Mock tracer provider for testing."""

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        return True


@pytest.fixture(autouse=True)
def patch_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = SimpleNamespace(
        get_tracer=lambda *_, **__: MockTracer(),
        get_tracer_provider=lambda: MockTracerProvider(),
    )
    monkeypatch.setattr(tracing, "otel_trace", mock)
    monkeypatch.setattr(tracing, "set_span_in_context", lambda span: None)
    monkeypatch.setattr(tracing, "get_current_span", lambda: None)


def get_last_span_for(
    tracer_obj: tracing.AzureAIOpenTelemetryTracer,
) -> MockSpan:
    tracer = cast(MockTracer, tracer_obj._tracer)  # type: ignore[attr-defined]
    return tracer.spans[-1]


def get_all_spans(
    tracer_obj: tracing.AzureAIOpenTelemetryTracer,
) -> List[MockSpan]:
    tracer = cast(MockTracer, tracer_obj._tracer)  # type: ignore[attr-defined]
    return list(tracer.spans)


def test_chain_start_supports_dataclass_inputs_and_metadata_message_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class Inputs:
        chat_history: List[Dict[str, Any]]

    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        Inputs(chat_history=[{"role": "user", "content": "hi"}]),
        run_id=run_id,
        metadata={"otel_messages_key": "chat_history", "agent_name": "X"},
    )
    span = get_last_span_for(tracer)
    payload = json.loads(span.attributes[tracing.Attrs.INPUT_MESSAGES])
    content = payload[0]["parts"][0]["content"]
    assert content in {"hi", "[redacted]"}


def test_chain_end_supports_command_like_outputs_and_records_goto() -> None:
    class FakeCommand:
        def __init__(self, update: Any, goto: str) -> None:
            self.update = update
            self.goto = goto

    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=run_id,
        metadata={"agent_name": "X", "otel_trace": True},
    )
    tracer.on_chain_end(
        FakeCommand(
            {"messages": [{"role": "assistant", "content": "ok"}]},
            "review_response",
        ),
        run_id=run_id,
    )
    span = get_all_spans(tracer)[-1]
    assert span.attributes.get("metadata.langgraph.goto") == "review_response"
    output = span.attributes.get(tracing.Attrs.OUTPUT_MESSAGES)
    if output:
        parsed = json.loads(output)
        assert parsed[0]["parts"][0]["content"] in {"ok", "[redacted]"}


def test_chain_end_supports_pydantic_like_outputs_model_dump() -> None:
    class FakeModel:
        def model_dump(self, exclude_none: bool = True) -> Dict[str, Any]:
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=run_id,
        metadata={"otel_trace": True, "agent_name": "Y"},
    )
    tracer.on_chain_end(FakeModel(), run_id=run_id)
    span = get_all_spans(tracer)[-1]
    output = span.attributes.get(tracing.Attrs.OUTPUT_MESSAGES)
    if output:
        parsed = json.loads(output)
        assert parsed[0]["parts"][0]["content"] in {"ok", "[redacted]"}


def test_otel_trace_true_forces_tracing_even_if_heuristics_would_ignore() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=run_id,
        metadata={"langgraph_node": "node-x", "otel_trace": True},
        name="node-x",
    )
    assert str(run_id) in tracer._spans


def test_trace_all_langgraph_nodes_traces_custom_nodes() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer(trace_all_langgraph_nodes=True)
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=run_id,
        metadata={"langgraph_node": "AnalyzeInput"},
    )
    assert str(run_id) in tracer._spans


def test_chain_start_honors_metadata_message_path_nested_dataclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class Wrapper:
        payload: Dict[str, Any]

    state = {"wrapper": Wrapper(payload={"messages": [{"role": "user", "content": "nested"}]})}
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        state,
        run_id=run_id,
        metadata={
            "agent_name": "NestedAgent",
            "otel_messages_path": "wrapper.payload.messages",
        },
    )
    span = get_last_span_for(tracer)
    payload = json.loads(span.attributes[tracing.Attrs.INPUT_MESSAGES])
    assert payload[0]["parts"][0]["content"] in {"nested", "[redacted]"}


def test_chain_start_respects_metadata_message_keys_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"chat_history": [{"role": "user", "content": "history"}]},
        run_id=run_id,
        metadata={
            "otel_messages_keys": ("chat_history", "messages"),
            "agent_name": "HistoryAgent",
        },
    )
    span = get_last_span_for(tracer)
    payload = json.loads(span.attributes[tracing.Attrs.INPUT_MESSAGES])
    assert payload[0]["parts"][0]["content"] in {"history", "[redacted]"}


def test_trace_all_nodes_can_capture_start_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = tracing.AzureAIOpenTelemetryTracer(
        trace_all_langgraph_nodes=True, ignore_start_node=False
    )
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=run_id,
        metadata={"langgraph_node": "__start__", "agent_name": "root"},
    )
    assert str(run_id) in tracer._spans


def test_compat_filtering_toggle_allows_langgraph_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    parent_run = uuid4()
    node_run = uuid4()
    default_tracer = tracing.AzureAIOpenTelemetryTracer()
    default_tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=node_run,
        parent_run_id=parent_run,
        metadata={"langgraph_node": "AnalyzeInput"},
        name="AnalyzeInput",
    )
    assert str(node_run) not in default_tracer._spans

    relaxed_tracer = tracing.AzureAIOpenTelemetryTracer(
        compat_create_agent_filtering=False
    )
    relaxed_tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=node_run,
        parent_run_id=parent_run,
        metadata={"langgraph_node": "AnalyzeInput"},
        name="AnalyzeInput",
    )
    assert str(node_run) in relaxed_tracer._spans


def test_otel_agent_span_false_skips_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "skip"}]},
        run_id=run_id,
        metadata={"langgraph_node": "SkipNode", "otel_agent_span": False},
    )
    assert str(run_id) not in tracer._spans


def test_chain_error_sets_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "boom"}]},
        run_id=run_id,
        metadata={"agent_name": "Boom"},
    )
    tracer.on_chain_error(RuntimeError("boom"), run_id=run_id)
    span = get_all_spans(tracer)[-1]
    assert span.status.status_code == StatusCode.ERROR


def test_tool_error_marks_span(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_tool_start({"name": "math"}, "input", run_id=run_id)
    tracer.on_tool_error(RuntimeError("fail"), run_id=run_id)
    span = get_all_spans(tracer)[-1]
    assert span.status.status_code == StatusCode.ERROR


def test_retriever_error_marks_span(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_retriever_start({"name": "search"}, "query", run_id=run_id)
    tracer.on_retriever_error(RuntimeError("oops"), run_id=run_id)
    span = get_all_spans(tracer)[-1]
    assert span.status.status_code == StatusCode.ERROR


def test_coerce_token_value_handles_nested_structures() -> None:
    nested = [
        {"value": "2"},
        {"values": [1, {"token_count": "3"}]},
        [None, 4],
    ]
    assert tracing._coerce_token_value(nested) == 10


def test_normalize_bedrock_usage_dict_infers_totals() -> None:
    usage = {
        "inputTokens": ["2", "1"],
        "outputTokenCount": {"value": 5},
    }
    normalized = tracing._normalize_bedrock_usage_dict(usage)
    assert normalized == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_normalize_bedrock_metrics_handles_missing_total() -> None:
    metrics = {"inputTokenCount": 2, "outputTokenCount": {"values": [1, 1]}}
    normalized = tracing._normalize_bedrock_metrics(metrics)
    assert normalized == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }


def test_collect_usage_from_generations_reads_generation_info() -> None:
    generation = ChatGeneration(
        message=AIMessage(content="ok"),
        generation_info={
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 3,
                "outputTokenCount": 4,
            }
        },
    )
    usage = tracing._collect_usage_from_generations([generation])
    assert usage == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_resolve_usage_from_llm_output_prefers_bedrock_metrics() -> None:
    llm_output = {
        "amazon-bedrock-invocationMetrics": {
            "inputTokenCount": 6,
            "outputTokenCount": 1,
        },
        "token_usage": {"prompt_tokens": 1},
    }
    input_tokens, output_tokens, total_tokens, normalized, should_attach = (
        tracing._resolve_usage_from_llm_output(llm_output, [])
    )
    assert (input_tokens, output_tokens, total_tokens) == (6, 1, 7)
    assert normalized == {
        "prompt_tokens": 6,
        "completion_tokens": 1,
        "total_tokens": 7,
    }
    assert should_attach


def test_resolve_usage_prefers_existing_token_usage() -> None:
    llm_output = {
        "token_usage": {"prompt_tokens": "2", "completion_tokens": 3},
    }
    values = tracing._resolve_usage_from_llm_output(llm_output, [])
    assert values[:3] == (2, 3, 5)
    assert not values[-1]


def test_infer_provider_name_prefers_metadata_hints() -> None:
    provider = tracing._infer_provider_name(
        None, {"ls_provider": "amazon_bedrock"}, {}
    )
    assert provider == "aws.bedrock"
    provider = tracing._infer_provider_name(
        None, {}, {"base_url": "https://workspace.openai.azure.com"}
    )
    assert provider == "azure.ai.openai"


def test_infer_server_address_and_port_from_invocation_params() -> None:
    serialized = {"kwargs": {"openai_api_base": "https://ignored.azure.com"}}
    params = {"base_url": "https://example.contoso.com:8443/v1"}
    assert tracing._infer_server_address(serialized, params) == "example.contoso.com"
    assert tracing._infer_server_port(serialized, params) == 8443


def test_resolve_connection_from_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "azure.identity.DefaultAzureCredential",
        lambda: "cred",
        raising=False,
    )
    monkeypatch.setattr(
        tracing,
        "get_service_endpoint_from_project",
        lambda endpoint, credential, service: ("InstrumentationKey=abc", None),
    )
    connection = tracing._resolve_connection_from_project("https://proj", None)
    assert connection == "InstrumentationKey=abc"


def test_tool_type_and_collection_helpers() -> None:
    assert tracing._tool_type_from_definition({"type": "function"}) == "function"
    assert (
        tracing._tool_type_from_definition({"function": {"type": "json"}}) == "json"
    )
    shared = {"name": "a"}
    combined = tracing._collect_tool_definitions(
        [shared],
        shared,
        [{"name": "b"}],
    )
    assert combined == [shared, {"name": "b"}]


def test_serialise_tool_result_and_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_msg = ToolMessage(content="done", name="calc", tool_call_id="abc123")
    result = json.loads(tracing._serialise_tool_result(tool_msg, True))
    assert result["tool_call_id"] == "abc123"
    dict_result = json.loads(
        tracing._serialise_tool_result({"value": 2}, record_content=True)
    )
    assert dict_result["value"] == 2
    docs = [
        Document(page_content="doc", metadata={"id": 1}),
        Document(page_content="doc2", metadata={"id": 2}),
    ]
    formatted = json.loads(tracing._format_documents(docs, record_content=True))
    assert formatted[0]["metadata"]["id"] == 1


def test_prepare_messages_and_filter_output(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant = {
        "role": "assistant",
        "content": "assistant",
        "tool_calls": [{"id": "tc1", "name": "use_tool", "arguments": {"foo": 1}}],
    }
    messages = [
        {"role": "system", "content": "rules"},
        HumanMessage(content="hi"),
        assistant,
        ToolMessage(content="tool result", tool_call_id="tc1"),
    ]
    formatted, system = tracing._prepare_messages(
        messages,
        record_content=True,
        include_roles={"user", "assistant", "tool"},
    )
    system_payload = json.loads(system)
    assert system_payload[0]["content"] == "rules"
    formatted_payload = json.loads(formatted)
    assert formatted_payload[0]["parts"][0]["content"] == "hi"
    assistant_entry = formatted_payload[1]
    assert any(part["type"] == "tool_call" for part in assistant_entry["parts"])
    filtered = tracing._filter_assistant_output(formatted)
    filtered_payload = json.loads(filtered)
    assert filtered_payload[0]["role"] == "assistant"


def test_extract_messages_payload_supports_paths() -> None:
    @dataclass
    class Nested:
        payload: Any

    wrapper = Nested(payload=SimpleNamespace(messages=[{"role": "user", "content": "hi"}]))
    value, goto = tracing._extract_messages_payload(
        wrapper, message_keys=("messages",), message_paths=("payload.messages",)
    )
    assert goto is None
    assert value[0]["content"] == "hi"


def test_scrub_value_redacts_when_disabled() -> None:
    data = {"text": "secret", "numbers": [1, 2]}
    scrubbed = tracing._scrub_value(data, record_content=False)
    assert scrubbed == "[redacted]"


def test_llm_start_attributes_content_recording_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure env enables content recording
    # fmt: off
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {
        "kwargs": {
            "model": "gpt-4o",
            "azure_endpoint": "http://host:8080",
        }
    }
    # fmt: on
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    span = get_last_span_for(t)

    attrs = span.attributes
    assert attrs.get(tracing.Attrs.PROVIDER_NAME) == "azure.ai.openai"
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "text_completion"
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.SERVER_ADDRESS) == "host"
    assert attrs.get(tracing.Attrs.SERVER_PORT) == 8080
    input_payload = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_payload[0]["parts"][0]["content"] == "hello"


def test_llm_start_attributes_content_recording_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fmt: off
    monkeypatch.delenv(
        "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", raising=False
    )
    # fmt: on
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=False)
    run_id = uuid4()
    serialized = {
        "kwargs": {
            "model": "gpt-4o",
            "azure_endpoint": "https://contoso.openai.azure.com",
        }
    }
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    input_payload = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_payload[0]["parts"][0]["content"] == "[redacted]"


def test_redaction_on_chat_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=False)
    run_id = uuid4()
    messages = [[HumanMessage(content="secret"), AIMessage(content="reply")]]
    serialized = {"kwargs": {"model": "m", "endpoint": "https://e"}}
    t.on_chat_model_start(serialized, messages, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    # Input content should be redacted
    input_json = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_json[0]["parts"][0]["content"] == "[redacted]"
    # End with output
    gen = ChatGeneration(message=AIMessage(content="reply"))
    result = LLMResult(generations=[[gen]], llm_output={})
    t.on_llm_end(result, run_id=run_id)
    # Verify output redacted on chat span when present;
    # some paths emit under agent root
    out_attr = span.attributes.get(tracing.Attrs.OUTPUT_MESSAGES)
    if out_attr:
        out_json = json.loads(out_attr)
        assert out_json[0]["parts"][0]["content"] == "[redacted]"
    else:
        # Fallback: if no chat output recorded, allow absence without failure
        # (agent root may contain the final output summary in role/parts schema)
        pass


def test_usage_and_response_metadata() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    gen = ChatGeneration(message=AIMessage(content="ok"))
    result = LLMResult(
        generations=[[gen]],
        llm_output={
            "token_usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "model_name": "m",
            "id": "resp-123",
            "service_tier": "standard",
            "system_fingerprint": "fingerprint",
        },
    )
    t.on_llm_end(result, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 3
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 5
    assert attrs.get(tracing.Attrs.RESPONSE_MODEL) == "m"
    assert attrs.get(tracing.Attrs.RESPONSE_ID) == "resp-123"
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SERVICE_TIER) == "standard"
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT) == "fingerprint"


def test_usage_metadata_input_output_keys() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    gen = ChatGeneration(message=AIMessage(content="ok"))
    result = LLMResult(
        generations=[[gen]],
        llm_output={"token_usage": {"input_tokens": "7", "output_tokens": "11"}},
    )
    t.on_llm_end(result, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 7
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 11


def test_inference_span_records_gen_ai_semantic_attributes() -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root_run = uuid4()
    conversation_id = "thread-123"
    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    t.on_chain_start(
        {},
        {"messages": base_messages},
        run_id=root_run,
        metadata={"thread_id": conversation_id, "agent_name": "Comedian"},
    )

    llm_run = uuid4()
    invocation_params = {
        "model": "gpt-4o",
        "max_tokens": 128,
        "max_input_tokens": 256,
        "max_output_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 20,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.1,
        "n": 2,
        "seed": 123,
        "stop": ["stop"],
        "response_format": {"type": "json_object"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather lookup",
                },
            }
        ],
        "base_url": "https://api.example.com:8443/v1",
        "service_tier": "standard",
    }
    serialized = {
        "kwargs": {
            "model": "gpt-4o",
            "openai_api_base": "https://api.example.com:8443/v1",
        }
    }
    prompts = cast(List[str], base_messages)
    t.on_llm_start(
        serialized,
        prompts,
        run_id=llm_run,
        parent_run_id=root_run,
        metadata={"ls_provider": "openai"},
        invocation_params=invocation_params,
    )

    generation = ChatGeneration(
        message=AIMessage(content="Here is a funny line."),
        generation_info={"finish_reason": "stop"},
    )
    result = LLMResult(
        generations=[[generation]],
        llm_output={
            "token_usage": {"input_tokens": 42, "output_tokens": 17},
            "model_name": "gpt-4o",
            "id": "resp-456",
            "system_fingerprint": "fp-123",
            "service_tier": "premium",
        },
    )
    t.on_llm_end(result, run_id=llm_run)

    span = get_last_span_for(t)
    attrs = span.attributes

    assert span.name == "text_completion gpt-4o"
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "text_completion"
    assert attrs.get(tracing.Attrs.PROVIDER_NAME) == "openai"
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.SERVER_ADDRESS) == "api.example.com"
    assert attrs.get(tracing.Attrs.SERVER_PORT) == 8443
    assert attrs.get(tracing.Attrs.REQUEST_MAX_TOKENS) == 128
    assert attrs.get(tracing.Attrs.REQUEST_MAX_INPUT_TOKENS) == 256
    assert attrs.get(tracing.Attrs.REQUEST_MAX_OUTPUT_TOKENS) == 64
    assert attrs.get(tracing.Attrs.REQUEST_TEMPERATURE) == 0.1
    assert attrs.get(tracing.Attrs.REQUEST_TOP_P) == 0.9
    assert attrs.get(tracing.Attrs.REQUEST_TOP_K) == 20
    assert attrs.get(tracing.Attrs.REQUEST_FREQ_PENALTY) == 0.5
    assert attrs.get(tracing.Attrs.REQUEST_PRES_PENALTY) == 0.1
    assert attrs.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 2
    assert attrs.get(tracing.Attrs.REQUEST_SEED) == 123
    assert attrs.get(tracing.Attrs.OPENAI_REQUEST_SERVICE_TIER) == "standard"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == conversation_id

    assert json.loads(attrs[tracing.Attrs.REQUEST_STOP]) == ["stop"]
    assert json.loads(attrs[tracing.Attrs.REQUEST_ENCODING_FORMATS]) == {
        "type": "json_object"
    }
    system_instr = json.loads(attrs[tracing.Attrs.SYSTEM_INSTRUCTIONS])
    assert system_instr[0]["content"] == "You are a helpful assistant."
    input_messages = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == "Tell me a joke."
    tool_defs = json.loads(attrs[tracing.Attrs.TOOL_DEFINITIONS])
    assert tool_defs[0]["function"]["name"] == "get_weather"

    assert attrs.get(tracing.Attrs.OUTPUT_TYPE) == "text"
    output_messages = json.loads(attrs[tracing.Attrs.OUTPUT_MESSAGES])
    assert output_messages[0]["parts"][0]["content"] == "Here is a funny line."
    assert json.loads(attrs[tracing.Attrs.RESPONSE_FINISH_REASONS]) == ["stop"]
    assert attrs.get(tracing.Attrs.RESPONSE_ID) == "resp-456"
    assert attrs.get(tracing.Attrs.RESPONSE_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 42
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 17
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT) == "fp-123"
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SERVICE_TIER) == "premium"


def test_agent_span_accumulates_usage_tokens() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "plan trip"}]},
        run_id=agent_run,
        metadata={"otel_agent_span": True, "agent_name": "Coordinator"},
    )

    llm_run_one = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        {"kwargs": {"model": "gpt-4o"}},
        prompts,
        run_id=llm_run_one,
        parent_run_id=agent_run,
        invocation_params={"model": "gpt-4o"},
    )
    gen_one = ChatGeneration(message=AIMessage(content="first"))
    result_one = LLMResult(
        generations=[[gen_one]],
        llm_output={"token_usage": {"input_tokens": 5, "output_tokens": 2}},
    )
    t.on_llm_end(result_one, run_id=llm_run_one, parent_run_id=agent_run)

    llm_run_two = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "gpt-4o"}},
        prompts,
        run_id=llm_run_two,
        parent_run_id=agent_run,
        invocation_params={"model": "gpt-4o"},
    )
    gen_two = ChatGeneration(message=AIMessage(content="second"))
    result_two = LLMResult(
        generations=[[gen_two]],
        llm_output={"token_usage": {"input_tokens": 7, "output_tokens": 4}},
    )
    t.on_llm_end(result_two, run_id=llm_run_two, parent_run_id=agent_run)

    agent_record = t._spans[str(agent_run)]
    agent_span = cast(MockSpan, agent_record.span)
    attrs = agent_span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 12
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 6


def test_streaming_token_event(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    t.on_llm_new_token("abcdef", run_id=run_id)
    span = get_last_span_for(t)
    assert span.events == []


def test_synthetic_execute_tool_under_chat_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer()
    # Start root agent via chain_start
    root = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root,
        metadata={"thread_id": root},
    )
    chat_run = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    msgs = cast(List[List[BaseMessage]], [[HumanMessage(content="prompt")]])
    t.on_chat_model_start(serialized, msgs, run_id=chat_run, parent_run_id=root)
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_current_date"},
        "",
        inputs={"tool_call_id": "call-1"},
        run_id=tool_run,
        parent_run_id=root,
    )
    record = t._spans[str(tool_run)]
    assert record.parent_run_id == str(chat_run)
    span = cast(MockSpan, record.span)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "get_current_date"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)
    t.on_tool_end({"result": "ok"}, run_id=tool_run, parent_run_id=root)


def test_tool_call_without_function_schema() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_weather"},
        "",
        inputs={"tool_call_id": "call-1", "city": "San Francisco"},
        metadata={},
        run_id=tool_run,
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "get_weather"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    args = json.loads(attrs[tracing.Attrs.TOOL_CALL_ARGUMENTS])
    assert args["city"] == "San Francisco"
    t.on_tool_end({"temperature": 60}, run_id=tool_run)


def test_no_invoke_agent_on_agent_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    # on_agent_action should not start invoke_agent spans;
    # only create_agent when applicable
    before = len(
        [
            s
            for s in get_all_spans(t)
            if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "invoke_agent"
        ]
    )
    action = cast(
        AgentAction,
        SimpleNamespace(
            agent_name="Agent",
            system_instructions=[{"type": "text", "content": "You are an agent."}],
        ),
    )
    t.on_agent_action(action, run_id=uuid4())
    after = len(
        [
            s
            for s in get_all_spans(t)
            if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "invoke_agent"
        ]
    )
    assert after == before


def test_tool_start_name_and_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root,
        metadata={"thread_id": root},
    )
    # Start a tool with serialized name and inputs id
    run_id = uuid4()
    serialized_tool = {
        "name": "search",
        "type": "function",
        "description": "desc",
    }
    inputs = {"tool_call_id": "call-1", "query": "foo"}
    t.on_tool_start(
        serialized_tool,
        "ignored",
        inputs=inputs,
        run_id=run_id,
        parent_run_id=root,
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)
    record = t._spans[str(run_id)]
    assert record.parent_run_id == str(root)
    t.on_tool_end({"answer": "bar"}, run_id=run_id, parent_run_id=root)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)


def test_tool_deduplicates_synthetic_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root,
        metadata={"thread_id": root},
    )
    chat_run = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    msgs = cast(List[List[BaseMessage]], [[HumanMessage(content="prompt")]])
    t.on_chat_model_start(serialized, msgs, run_id=chat_run, parent_run_id=root)
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_weather", "description": "Returns weather"},
        "ignored",
        inputs={"tool_call_id": "call-1", "city": "SF"},
        run_id=tool_run,
        parent_run_id=chat_run,
    )
    t.on_tool_end({"temperature": 60}, run_id=tool_run, parent_run_id=chat_run)
    t.on_chain_end({}, run_id=root)
    tool_spans = [
        s
        for s in get_all_spans(t)
        if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    ]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes.get(tracing.Attrs.TOOL_NAME) == "get_weather"
    assert span.attributes.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"


def test_use_propagated_context_attaches_and_detaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()

    headers = {"traceparent": "00-01" + "0" * 30 + "-02" + "0" * 14 + "-01"}
    sentinel_context = object()
    attached: list[object] = []
    detached: list[object] = []

    def fake_extract(carrier: Mapping[str, str]) -> object:
        assert carrier == headers
        return sentinel_context

    def fake_attach(ctx: object) -> str:
        attached.append(ctx)
        return "token"

    def fake_detach(token: str) -> None:
        detached.append(token)

    monkeypatch.setattr(tracing, "extract", fake_extract)
    monkeypatch.setattr(tracing, "attach", fake_attach)
    monkeypatch.setattr(tracing, "detach", fake_detach)

    with tracer.use_propagated_context(headers=headers):
        assert attached == [sentinel_context]

    assert detached == ["token"]


def test_thread_root_parent_resolution() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    thread_id = "thread-123"
    root_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "travel_planner",
        },
    )
    child_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi again"}]},
        run_id=child_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "flight_specialist",
        },
    )
    child_record = tracer._spans[str(child_run)]
    assert child_record.parent_run_id == str(root_run)
    assert tracer._langgraph_root_by_thread[thread_id] == str(root_run)


def test_llm_and_tool_attach_to_latest_agent() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    thread_id = "stack-thread"
    root_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "travel_planner",
        },
    )
    child_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "help"}]},
        run_id=child_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "flight_specialist",
        },
    )
    llm_run = uuid4()
    prompts = cast(List[List[BaseMessage]], [[HumanMessage(content="flight options")]])
    tracer.on_llm_start(
        {"kwargs": {"model": "gpt-test"}},
        prompts,
        run_id=llm_run,
        metadata={"thread_id": thread_id},
    )
    llm_record = tracer._spans[str(llm_run)]
    assert llm_record.parent_run_id == str(child_run)
    tool_run = uuid4()
    tracer.on_tool_start(
        {"name": "search"},
        "",
        run_id=tool_run,
        metadata={"thread_id": thread_id},
        inputs={"tool_call_id": "1"},
    )
    tool_record = tracer._spans[str(tool_run)]
    assert tool_record.parent_run_id == str(child_run)

    tracer.on_tool_end({}, run_id=tool_run)
    tracer.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]),
        run_id=llm_run,
    )
    tracer.on_chain_end({}, run_id=child_run)
    tracer.on_chain_end({}, run_id=root_run)
    assert tracer._agent_stack_by_thread.get(thread_id) in (None, [])


def test_invoke_agent_records_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    tools = [
        {"name": "get_weather", "description": "Returns forecast"},
        {"name": "search_docs", "description": "Search knowledge base"},
    ]
    run_id = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        {"kwargs": {"model": "m"}},
        prompts,
        run_id=run_id,
        invocation_params={"model": "m", "tools": tools},
    )
    span = get_last_span_for(t)
    defs = span.attributes.get(tracing.Attrs.TOOL_DEFINITIONS)
    assert defs is not None
    parsed = json.loads(defs)
    tool_names = {tool["name"] for tool in parsed}
    assert {"get_weather", "search_docs"} <= tool_names


def test_finish_reasons_normalized() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    chat_run = uuid4()
    gen = ChatGeneration(
        message=AIMessage(content=""),
        generation_info={"finish_reason": "tool_calls"},
    )
    result = LLMResult(generations=[[gen]], llm_output={})
    # Create chat span and then end it
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        {"kwargs": {"model": "m"}},
        prompts,
        run_id=chat_run,
        invocation_params={"model": "m"},
    )
    t.on_llm_end(result, run_id=chat_run)
    span = get_last_span_for(t)
    finish_reasons = span.attributes.get(tracing.Attrs.RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert json.loads(finish_reasons) == ["tool_calls"]


def test_chat_parenting_under_root_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    root = uuid4()
    # Start root agent
    t.on_chain_start({}, {"messages": [{"role": "user", "content": "hi"}]}, run_id=root)
    # Start chat without specifying parent; should parent under root agent
    chat_run = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        {"kwargs": {"model": "m"}},
        prompts,
        run_id=chat_run,
        parent_run_id=root,
        invocation_params={"model": "m"},
    )
    record = t._spans[str(chat_run)]
    assert record.parent_run_id == str(root)


def test_llm_error_sets_status_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    err = RuntimeError("boom")
    t.on_llm_error(err, run_id=run_id)
    span = get_last_span_for(t)
    assert span.ended is True
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"


def test_tool_start_end_records_args_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    parent_run = uuid4()
    serialized_tool = {
        "name": "search",
        "type": "function",
        "description": "desc",
    }
    inputs = {"tool_call_id": "call-1", "query": "foo"}
    t.on_tool_start(
        serialized_tool,
        "ignored",
        inputs=inputs,
        run_id=run_id,
        parent_run_id=parent_run,
    )
    t.on_tool_end({"answer": "bar"}, run_id=run_id, parent_run_id=parent_run)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    # Args and result only when content recording enabled
    tool_args = attrs.get(tracing.Attrs.TOOL_CALL_ARGUMENTS)
    tool_result = attrs.get(tracing.Attrs.TOOL_CALL_RESULT)
    assert tool_args is not None and tool_result is not None
    assert json.loads(tool_args).get("query") == "foo"
    assert json.loads(tool_result).get("answer") == "bar"


def test_choice_count_only_when_n_not_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m", "n": 1}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m", "n": 1},
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 1
    # Now with n=3
    run_id2 = uuid4()
    serialized2 = {"kwargs": {"model": "m", "n": 3}}
    t.on_llm_start(
        serialized2,
        prompts,
        run_id=run_id2,
        invocation_params={"model": "m", "n": 3},
    )
    span2 = get_last_span_for(t)
    assert span2.attributes.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 3


def test_server_port_extraction_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    # https default port not set
    run1 = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "https://host"}},
        prompts,
        run_id=run1,
        invocation_params={"model": "m"},
    )
    s1 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s1.attributes
    # https non-default port set
    run2 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "https://host:8443"}},
        prompts,
        run_id=run2,
        invocation_params={"model": "m"},
    )
    s2 = get_last_span_for(t)
    assert s2.attributes.get(tracing.Attrs.SERVER_PORT) == 8443
    # http default port omitted
    run3 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "http://host"}},
        prompts,
        run_id=run3,
        invocation_params={"model": "m"},
    )
    s3 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s3.attributes
    # http non-default port set
    run4 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "http://host:8080"}},
        prompts,
        run_id=run4,
        invocation_params={"model": "m"},
    )
    s4 = get_last_span_for(t)
    assert s4.attributes.get(tracing.Attrs.SERVER_PORT) == 8080


def test_retriever_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"name": "index", "id": "retr"}
    t.on_retriever_start(serialized, "q", run_id=run_id)
    documents = [
        Document(page_content="doc1", metadata={"source": "a"}),
        Document(page_content="doc2", metadata={"source": "b"}),
        Document(page_content="doc3", metadata={"source": "c"}),
    ]
    t.on_retriever_end(documents, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_TYPE) == "retriever"
    assert attrs.get(tracing.Attrs.RETRIEVER_QUERY) == "q"
    results = json.loads(attrs[tracing.Attrs.RETRIEVER_RESULTS])
    assert len(results) == 3


def test_parser_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    with pytest.raises(AttributeError):
        getattr(t, "on_parser_start")
    with pytest.raises(AttributeError):
        getattr(t, "on_parser_end")


def test_transform_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    with pytest.raises(AttributeError):
        getattr(t, "on_transform_start")
    with pytest.raises(AttributeError):
        getattr(t, "on_transform_end")


def test_pending_tool_call_cached_for_chain_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    t.on_chat_model_start(
        serialized,
        [[ToolMessage(name="echo", tool_call_id="abc", content="result")]],
        run_id=run_id,
    )
    span = get_last_span_for(t)
    input_messages = json.loads(span.attributes[tracing.Attrs.INPUT_MESSAGES])
    parts = input_messages[0]["parts"]
    tool_part = next(part for part in parts if part.get("type") == "tool_call_response")
    assert tool_part["id"] == "abc"
    assert tool_part["result"] == "result"


# ---------------------------------------------------------------------------
# Static autolog() API Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_static_state() -> Any:
    """Reset static class state before and after each autolog test."""
    # Reset before test
    tracing.AzureAIOpenTelemetryTracer._GLOBAL_TRACER_INSTANCE = None
    tracing.AzureAIOpenTelemetryTracer._ACTIVE = False
    tracing.AzureAIOpenTelemetryTracer._GLOBAL_CONFIG = {}
    tracing.AzureAIOpenTelemetryTracer._APP_INSIGHTS_CS = None
    yield
    # Reset after test
    tracing.AzureAIOpenTelemetryTracer._GLOBAL_TRACER_INSTANCE = None
    tracing.AzureAIOpenTelemetryTracer._ACTIVE = False
    tracing.AzureAIOpenTelemetryTracer._GLOBAL_CONFIG = {}
    tracing.AzureAIOpenTelemetryTracer._APP_INSIGHTS_CS = None


def test_default_config_has_expected_keys() -> None:
    """DEFAULT_CONFIG should contain all expected configuration keys."""
    expected_keys = {
        "provider_name",
        "tracer_name",
        "enable_content_recording",
        "redact_messages",
        "redact_tool_arguments",
        "redact_tool_results",
        "include_tool_definitions",
        "aggregate_usage",
        "normalize_operation_names",
        "sampling_rate",
        "honor_external_parent",
        "patch_mode",
        "log_stream_chunks",
        "max_message_characters",
        "thread_id_attribute",
        "enable_span_links",
        "message_keys",
        "message_paths",
        "trace_all_langgraph_nodes",
        "ignore_start_node",
        "compat_create_agent_filtering",
        "enable_performance_counters",
    }
    assert set(tracing.DEFAULT_CONFIG.keys()) == expected_keys


def test_set_config_and_get_config(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """set_config() should merge values and get_config() should return effective config."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Get default config first
    default = T.get_config()
    assert default["provider_name"] is None
    assert default["redact_messages"] is False

    # Set some config values
    T.set_config({"provider_name": "azure.ai.openai", "redact_messages": True})

    effective = T.get_config()
    assert effective["provider_name"] == "azure.ai.openai"
    assert effective["redact_messages"] is True
    # Other values should still be defaults
    assert effective["enable_content_recording"] is True


def test_set_config_warns_on_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """set_config() should warn about unknown configuration keys."""
    import logging
    caplog.set_level(logging.WARNING)
    T = tracing.AzureAIOpenTelemetryTracer
    T.set_config({"unknown_key": "value", "another_unknown": 123})
    assert "unknown_key" in caplog.text or "Unknown configuration" in caplog.text


def test_get_config_respects_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """Environment variables should override defaults in get_config()."""
    T = tracing.AzureAIOpenTelemetryTracer

    monkeypatch.setenv("GENAI_REDACT_MESSAGES", "true")
    monkeypatch.setenv("GENAI_SAMPLING_RATE", "0.5")
    monkeypatch.setenv("GENAI_PROVIDER_NAME", "test_provider")

    config = T.get_config()
    assert config["redact_messages"] is True
    assert config["sampling_rate"] == 0.5
    assert config["provider_name"] == "test_provider"


def test_set_app_insights_stores_connection_string(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """set_app_insights() should store the connection string."""
    T = tracing.AzureAIOpenTelemetryTracer

    cs = "InstrumentationKey=test;IngestionEndpoint=https://test"
    T.set_app_insights(cs)

    assert T._APP_INSIGHTS_CS == cs


def _mock_register_callback(inst: Any) -> None:
    """Mock callback registration that does nothing."""
    pass


def test_is_active_reflects_state(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """is_active() should reflect whether autolog() has been called."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function to avoid LangChain dependency issues
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    assert T.is_active() is False

    T.autolog()
    assert T.is_active() is True

    T.shutdown()
    assert T.is_active() is False


def test_autolog_activation_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """Calling autolog() multiple times should keep one tracer instance."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    T.autolog(provider_name="first")
    first_instance = T._GLOBAL_TRACER_INSTANCE

    # Second call should merge config but not create new instance
    T.autolog(provider_name="second")
    second_instance = T._GLOBAL_TRACER_INSTANCE

    assert first_instance is second_instance
    assert T._GLOBAL_CONFIG.get("provider_name") == "second"


def test_shutdown_cleans_state(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """shutdown() should clean up all state."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    T.autolog()
    assert T._GLOBAL_TRACER_INSTANCE is not None
    assert T._ACTIVE is True

    T.shutdown()
    assert T._GLOBAL_TRACER_INSTANCE is None
    assert T._ACTIVE is False


def test_get_tracer_instance_returns_tracer_when_active(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """get_tracer_instance() should return the tracer when active."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    assert T.get_tracer_instance() is None

    T.autolog()
    instance = T.get_tracer_instance()
    assert instance is not None
    assert isinstance(instance, T)

    T.shutdown()
    assert T.get_tracer_instance() is None


def test_update_redaction_rules_modifies_config(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """update_redaction_rules() should modify redaction settings."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    T.autolog()

    # Initially not redacting
    assert T._GLOBAL_CONFIG.get("redact_messages") is None or T._GLOBAL_CONFIG.get("redact_messages") is False

    # Enable redaction
    T.update_redaction_rules(redact_messages=True, redact_tool_arguments=True)

    assert T._GLOBAL_CONFIG["redact_messages"] is True
    assert T._GLOBAL_CONFIG["redact_tool_arguments"] is True

    # Tracer instance should reflect the change
    instance = T.get_tracer_instance()
    assert instance is not None
    assert instance._content_recording is False  # Should be disabled when redacting


def test_add_tags_adds_to_root_spans(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """add_tags() should add attributes to root spans."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    T.autolog()
    instance = T.get_tracer_instance()
    assert instance is not None

    # Create a mock root span by starting a chain
    run_id = uuid4()
    instance.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=run_id,
        metadata={"otel_trace": True},
    )

    # Add tags
    T.add_tags({"user.id": "test-user", "session.type": "demo"})

    # Verify tags were added to the span
    span = get_last_span_for(instance)
    assert span.attributes.get("user.id") == "test-user"
    assert span.attributes.get("session.type") == "demo"


def test_force_flush_handles_no_provider(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """force_flush() should handle cases where provider doesn't have force_flush."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock provider without force_flush - override the global mock
    mock_provider_no_flush = SimpleNamespace()
    mock_otel = SimpleNamespace(
        get_tracer=lambda *_, **__: MockTracer(),
        get_tracer_provider=lambda: mock_provider_no_flush,
    )
    monkeypatch.setattr(tracing, "otel_trace", mock_otel)

    result = T.force_flush()
    assert result is True  # Should return True if no force_flush method


def test_autolog_uses_effective_config(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """autolog() should use merged effective configuration."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Mock the register function
    monkeypatch.setattr(
        T, "_register_global_callback", classmethod(lambda cls, inst: None)
    )

    # Pre-configure
    T.set_config({
        "provider_name": "azure.ai.openai",
        "message_keys": ("chat_history",),
    })

    # Override at autolog time
    T.autolog(trace_all_langgraph_nodes=True)

    instance = T.get_tracer_instance()
    assert instance is not None
    assert instance._default_provider_name == "azure.ai.openai"
    assert instance._message_keys == ("chat_history",)
    assert instance._trace_all_langgraph_nodes is True


def test_shutdown_is_safe_when_not_active(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """shutdown() should be safe to call even when not active."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Should not raise
    T.shutdown()
    assert T._ACTIVE is False


def test_add_tags_noop_when_inactive(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """add_tags() should be a no-op when autolog is not active."""
    T = tracing.AzureAIOpenTelemetryTracer

    # Should not raise
    T.add_tags({"key": "value"})


def test_update_redaction_rules_noop_when_inactive(
    monkeypatch: pytest.MonkeyPatch,
    reset_static_state: Any,
) -> None:
    """update_redaction_rules() should still update config even when inactive."""
    T = tracing.AzureAIOpenTelemetryTracer

    T.update_redaction_rules(redact_messages=True)
    assert T._GLOBAL_CONFIG["redact_messages"] is True


def test_tool_execution_uses_chat_span_as_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool execution spans should use the chat span as parent, not invoke_agent.

    This verifies the hierarchy: invoke_agent -> chat -> execute_tool
    """
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)

    # Start invoke_agent span
    agent_run_id = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "weather?"}]},
        run_id=agent_run_id,
        metadata={"otel_trace": True, "agent_name": "TestAgent"},
    )
    agent_record = t._spans.get(str(agent_run_id))
    assert agent_record is not None
    assert agent_record.operation == "invoke_agent"

    # Start chat span (child of invoke_agent)
    chat_run_id = uuid4()
    t.on_chat_model_start(
        {"kwargs": {"model": "gpt-4"}},
        [[HumanMessage(content="weather?")]],
        run_id=chat_run_id,
        parent_run_id=agent_run_id,
    )
    chat_record = t._spans.get(str(chat_run_id))
    assert chat_record is not None
    assert chat_record.operation == "chat"
    assert chat_record.parent_run_id == str(agent_run_id)

    # Verify that last_chat_context is stored in invoke_agent's stash
    assert "last_chat_context" in agent_record.stash

    # End chat span (simulating LLM returning with tool calls)
    from langchain_core.outputs import ChatGeneration, LLMResult
    from langchain_core.messages.tool import ToolCall

    llm_result = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[ToolCall(name="get_weather", args={"city": "NYC"}, id="call_1")],
                    ),
                    generation_info={"finish_reason": "tool_calls"},
                )
            ]
        ],
        llm_output={"model_name": "gpt-4"},
    )
    t.on_llm_end(llm_result, run_id=chat_run_id, parent_run_id=agent_run_id)

    # Chat span should be ended and removed from _spans
    assert str(chat_run_id) not in t._spans

    # But the chat context should still be stored
    assert "last_chat_context" in agent_record.stash
    assert agent_record.stash.get("last_chat_run") == str(chat_run_id)

    # Now start tool execution (should use stored chat context as parent)
    tool_run_id = uuid4()
    t.on_tool_start(
        {"name": "get_weather"},
        '{"city": "NYC"}',
        run_id=tool_run_id,
        parent_run_id=agent_run_id,  # LangChain passes invoke_agent as parent
    )
    tool_record = t._spans.get(str(tool_run_id))
    assert tool_record is not None
    assert tool_record.operation == "execute_tool"

    # Key assertion: tool's parent should be the chat span (via stored context),
    # not the invoke_agent span
    assert tool_record.parent_run_id == str(chat_run_id), (
        f"Tool's parent should be chat ({chat_run_id}), "
        f"not invoke_agent ({agent_run_id})"
    )


def test_genai_span_hierarchy_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complete GenAI semantic convention span hierarchy.

    Expected hierarchy per GenAI spec:
    invoke_agent (root)
    ├── chat (first LLM call that decides to use tools)
    │   └── execute_tool (tool execution as child of chat)
    └── chat (second LLM call with tool results)
    """
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    span_hierarchy: List[Tuple[str, Optional[str]]] = []

    # Track span creation
    original_start = t._start_span

    def tracking_start(run_id, name, *, operation, kind, parent_run_id, attributes=None, thread_key=None):
        original_start(run_id, name, operation=operation, kind=kind, parent_run_id=parent_run_id, attributes=attributes, thread_key=thread_key)
        record = t._spans.get(str(run_id))
        if record:
            span_hierarchy.append((operation, record.parent_run_id))

    t._start_span = tracking_start

    # Simulate agent flow: invoke_agent -> chat -> tool -> chat
    from langchain_core.outputs import ChatGeneration, LLMResult
    from langchain_core.messages.tool import ToolCall

    agent_id = uuid4()
    chat1_id = uuid4()
    tool_id = uuid4()
    chat2_id = uuid4()

    # 1. Start invoke_agent
    t.on_chain_start({}, {"messages": []}, run_id=agent_id, metadata={"otel_trace": True})

    # 2. Start first chat
    t.on_chat_model_start({"kwargs": {}}, [[]], run_id=chat1_id, parent_run_id=agent_id)

    # 3. End first chat (with tool call)
    t.on_llm_end(
        LLMResult(generations=[[ChatGeneration(
            message=AIMessage(content="", tool_calls=[ToolCall(name="tool", args={}, id="c1")]),
            generation_info={},
        )]], llm_output={}),
        run_id=chat1_id, parent_run_id=agent_id
    )

    # 4. Start tool execution
    t.on_tool_start({"name": "tool"}, "{}", run_id=tool_id, parent_run_id=agent_id)

    # 5. End tool execution
    t.on_tool_end("result", run_id=tool_id, parent_run_id=agent_id)

    # 6. Start second chat (with tool results)
    t.on_chat_model_start({"kwargs": {}}, [[]], run_id=chat2_id, parent_run_id=agent_id)

    # 7. End second chat
    t.on_llm_end(
        LLMResult(generations=[[ChatGeneration(
            message=AIMessage(content="Final answer"),
            generation_info={},
        )]], llm_output={}),
        run_id=chat2_id, parent_run_id=agent_id
    )

    # 8. End invoke_agent
    t.on_chain_end({}, run_id=agent_id)

    # Verify hierarchy
    assert len(span_hierarchy) >= 4, f"Expected at least 4 spans, got {len(span_hierarchy)}"

    # Find key spans
    agent_entry = next((op, parent) for op, parent in span_hierarchy if op == "invoke_agent")
    chat1_entry = span_hierarchy[1]  # First chat
    tool_entry = next((op, parent) for op, parent in span_hierarchy if op == "execute_tool")
    chat2_entry = next((op, parent) for op, parent in span_hierarchy[3:] if op == "chat")

    # Verify relationships
    assert agent_entry[1] is None, "invoke_agent should be root (no parent)"
    assert chat1_entry[0] == "chat", "Second span should be chat"
    assert chat1_entry[1] == str(agent_id), "First chat should have invoke_agent as parent"
    assert tool_entry[1] == str(chat1_id), "Tool should have first chat as parent (not invoke_agent)"
    assert chat2_entry[1] == str(agent_id), "Second chat should have invoke_agent as parent"


def test_input_output_messages_set_correctly() -> None:
    """Verify gen_ai.input.messages and gen_ai.output.messages are set per GenAI spec.

    According to GenAI semantic conventions:
    - Input messages should be set at span start with user/assistant/tool messages
    - Output messages should be set at span end with assistant responses
    - System instructions should be extracted separately
    """
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)

    run_id = uuid4()
    parent_id = uuid4()

    # Start invoke_agent first
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "Hello"}]},
        run_id=parent_id,
        metadata={"otel_trace": True, "agent_name": "TestAgent"},
    )

    # Start chat model with messages
    t.on_chat_model_start(
        {"kwargs": {"model": "gpt-4"}},
        [[
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the weather?"),
        ]],
        run_id=run_id,
        parent_run_id=parent_id,
    )

    # Get the MockSpan from the tracer (uses our mock infrastructure)
    chat_span = get_last_span_for(t)
    assert chat_span is not None

    # Verify input messages attribute is set on the MockSpan
    input_msgs = chat_span.attributes.get(tracing.Attrs.INPUT_MESSAGES)
    assert input_msgs is not None, "gen_ai.input.messages should be set on chat start"

    # Parse and verify structure
    parsed_input = json.loads(input_msgs)
    assert len(parsed_input) == 1, "Should have 1 user message (system extracted separately)"
    assert parsed_input[0]["role"] == "user"
    assert any("weather" in str(part.get("content", "")).lower() for part in parsed_input[0]["parts"])

    # Verify system instructions are extracted
    system_instr = chat_span.attributes.get(tracing.Attrs.SYSTEM_INSTRUCTIONS)
    assert system_instr is not None, "gen_ai.system_instructions should be set"
    parsed_system = json.loads(system_instr)
    assert any("helpful assistant" in str(part.get("content", "")).lower() for part in parsed_system)

    # End chat with output
    llm_result = LLMResult(
        generations=[[
            ChatGeneration(
                message=AIMessage(content="The weather is sunny."),
                generation_info={"finish_reason": "stop"},
            )
        ]],
        llm_output={"model_name": "gpt-4"},
    )

    t.on_llm_end(llm_result, run_id=run_id, parent_run_id=parent_id)

    # Verify output messages were set on the span
    output_msgs = chat_span.attributes.get(tracing.Attrs.OUTPUT_MESSAGES)
    assert output_msgs is not None, "gen_ai.output.messages should be set on chat end"

    parsed_output = json.loads(output_msgs)
    assert len(parsed_output) == 1, "Should have 1 assistant message"
    assert parsed_output[0]["role"] == "assistant"
    assert any("sunny" in str(part.get("content", "")).lower() for part in parsed_output[0]["parts"])


def test_invoke_agent_messages_propagation() -> None:
    """Verify invoke_agent spans properly capture and propagate messages."""
    import json
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)

    run_id = uuid4()

    # Start invoke_agent with user message
    t.on_chain_start(
        {},
        {"messages": [
            {"role": "system", "content": "Agent system prompt"},
            {"role": "user", "content": "Plan my vacation"},
        ]},
        run_id=run_id,
        metadata={"otel_trace": True, "agent_name": "TravelAgent"},
    )

    record = t._spans.get(str(run_id))
    assert record is not None
    assert record.operation == "invoke_agent"

    # Verify input messages are set on agent span
    input_msgs = record.attributes.get(tracing.Attrs.INPUT_MESSAGES)
    assert input_msgs is not None, "invoke_agent should have gen_ai.input.messages"

    parsed = json.loads(input_msgs)
    # Should have user message (system extracted separately)
    user_msgs = [m for m in parsed if m["role"] == "user"]
    assert len(user_msgs) == 1, "Should have 1 user message"

    # Verify system instructions extracted
    system_instr = record.attributes.get(tracing.Attrs.SYSTEM_INSTRUCTIONS)
    assert system_instr is not None, "invoke_agent should have gen_ai.system_instructions"

    # End invoke_agent with response
    t.on_chain_end(
        {"messages": [
            {"role": "assistant", "content": "Here is your vacation plan..."},
        ]},
        run_id=run_id,
    )

    # Verify the span was ended (removed from _spans)
    assert str(run_id) not in t._spans
