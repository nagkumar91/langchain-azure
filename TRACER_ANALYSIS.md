# Comprehensive Analysis of Tracer Code in LangChain Azure

## Executive Summary

The repository contains a sophisticated **AzureAIOpenTelemetryTracer** - a LangChain callback handler that emits OpenTelemetry (OTEL) spans compatible with the GenAI semantic conventions. It's designed to instrument LangChain and LangGraph applications with distributed tracing capabilities, enabling correlation with upstream HTTP traces through W3C traceparent headers and asyncio context propagation.

---

## 1. Location of Tracer Code

### Primary Source Files
```
/Users/nagkumar/Documents/msft.nosync/langchain-azure/
├── libs/azure-ai/langchain_azure_ai/callbacks/tracers/
│   ├── __init__.py                          # Exports AzureAIOpenTelemetryTracer
│   ├── inference_tracing.py                 # Main tracer implementation (2504 lines)
│   └── _semantic_conventions_gen_ai.py      # OpenTelemetry GenAI attribute constants
└── samples/
    └── tracer_compat_sample.py              # Sample usage with LangGraph
```

### Test Files
```
libs/azure-ai/tests/
├── unit_tests/
│   └── test_inference_tracing.py            # Unit tests for tracer helpers
└── integration_tests/tracers/
    ├── test_agent_tracing_e2e.py            # E2E integration tests (612 lines)
    ├── test_custom_graph_tracer_compat.py   # Custom compatibility tests
    ├── test_negative_agent_tracer.py        # Negative scenario tests
    └── README.md                             # Test documentation
```

---

## 2. Tracer Initialization

### Core Class: `AzureAIOpenTelemetryTracer`

**Location**: `inference_tracing.py`, lines 1092-1161

```python
class AzureAIOpenTelemetryTracer(BaseCallbackHandler):
    """LangChain callback handler that emits OpenTelemetry GenAI spans."""
    
    _azure_monitor_configured: bool = False
    _configure_lock: Lock = Lock()
    _schema_url: str = Schemas.V1_28_0.value
```

### Initialization Signature

```python
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
    _prepare_messages_fn: Optional[Callable[..., tuple[Optional[str], Optional[str]]]] = None,
) -> None:
```

### Key Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `connection_string` | str | None | Azure Monitor connection string for telemetry export |
| `enable_content_recording` | bool | True | Record LLM inputs/outputs (False = redact) |
| `project_endpoint` | str | None | Azure AI project endpoint to resolve telemetry connection |
| `credential` | Any | None | Azure credential for project resolution |
| `name` | str | "AzureAIOpenTelemetryTracer" | Tracer instance name |
| `agent_id` | str | None | Default agent ID for all spans |
| `provider_name` | str | None | Default provider (azure.ai.openai, openai, etc.) |
| `message_keys` | Sequence[str] | ("messages",) | Keys to extract messages from inputs |
| `message_paths` | Sequence[str] | () | Dot-paths to extract messages (e.g., "payload.messages") |
| `trace_all_langgraph_nodes` | bool | False | Emit spans for all LangGraph nodes |
| `ignore_start_node` | bool | True | Skip `__start__` node spans |
| `compat_create_agent_filtering` | bool | True | Filter certain auto-generated agent spans |
| `_prepare_messages_fn` | Callable | None | Custom message formatter function |

### Internal State

```python
self._spans: Dict[str, _SpanRecord] = {}          # Active span records by run_id
self._lock = Lock()                                # Thread safety
self._ignored_runs: set[str] = set()               # Runs to skip
self._run_parent_override: Dict[str, Optional[str]] = {}  # Parent override mapping
self._langgraph_root_by_thread: Dict[str, str] = {}       # Thread to root span
self._agent_stack_by_thread: Dict[str, List[str]] = {}    # Agent call stack per thread
self._goto_parent_by_thread: Dict[str, List[str]] = {}    # LangGraph goto tracking
```

### Azure Monitor Configuration

```python
@classmethod
def _configure_azure_monitor(cls, connection_string: str) -> None:
    with cls._configure_lock:
        if cls._azure_monitor_configured:
            return
        configure_azure_monitor(connection_string=connection_string)
        cls._azure_monitor_configured = True
```

The tracer automatically configures Azure Monitor (singleton pattern) when a connection string is provided.

---

## 3. How Traceparent Works & Propagation

### W3C Trace Context Extraction

**Location**: `inference_tracing.py`, lines 233-268

```python
def _collect_trace_headers(
    mapping: Optional[Mapping[str, Any]],
) -> Optional[dict[str, str]]:
    if not mapping:
        return None
    headers: dict[str, str] = {}
    for key, value in mapping.items():
        key_lower = str(key).lower()
        if key_lower in {"traceparent", "tracestate"} and value:
            headers[key_lower] = str(value)
    return headers or None


def _extract_trace_headers(container: Any) -> Optional[dict[str, str]]:
    """Extract traceparent/tracestate from nested containers."""
    mapping = _to_mapping(container)
    headers = _collect_trace_headers(mapping)
    if headers:
        return headers
    if not mapping:
        return None
    # Try nested keys: headers, header, http_headers, request_headers, metadata, request
    _NESTED_HEADER_KEYS = (
        "headers", "header", "http_headers", "request_headers", "metadata", "request",
    )
    for key in _NESTED_HEADER_KEYS:
        nested = mapping.get(key)
        nested_mapping = _to_mapping(nested) if nested is not None else None
        headers = _collect_trace_headers(nested_mapping)
        if headers:
            return headers
    return None
```

### Context Propagation with `use_propagated_context`

**Location**: `inference_tracing.py`, lines 1162-1190

```python
@contextmanager
def use_propagated_context(
    self,
    *,
    headers: Mapping[str, str] | None,
) -> Iterator[None]:
    """Temporarily adopt an upstream trace context extracted from headers.
    
    This enables scenarios where an HTTP ingress or orchestrator wants to
    ensure the LangGraph spans are correlated with the inbound trace.
    """
    if not headers:
        yield
        return
    try:
        ctx = extract(headers)  # OpenTelemetry extract from W3C headers
    except Exception:
        LOGGER.debug(
            "Failed to extract OpenTelemetry context from headers; "
            "continuing without propagation.",
            exc_info=True,
        )
        yield
        return
    token = attach(ctx)         # Attach to current context
    try:
        yield
    finally:
        detach(token)           # Clean up after span creation
```

### Where Traceparent is Used

In `on_chain_start` (lines 1593-1644):

```python
trace_headers = (
    _extract_trace_headers(metadata)
    or _extract_trace_headers(inputs)
    or _extract_trace_headers(kwargs)
)

span_name = f"invoke_agent {attributes[Attrs.AGENT_NAME]}"
with self.use_propagated_context(headers=trace_headers):
    self._start_span(
        run_id,
        span_name,
        operation="invoke_agent",
        kind=SpanKind.CLIENT,
        parent_run_id=effective_parent_run_id,
        attributes=attributes,
        thread_key=thread_key,
    )
```

**Key Behavior**: 
- Extracts `traceparent` and `tracestate` headers from metadata, inputs, or kwargs
- Uses OpenTelemetry's `extract()` to deserialize W3C trace context
- Temporarily attaches the upstream context before creating spans
- Allows LangGraph spans to be correlated with HTTP ingress traces

---

## 4. Integration with LangChain Callback System

### Callback Handler Inheritance

```python
class AzureAIOpenTelemetryTracer(BaseCallbackHandler):
    """LangChain callback handler that emits OpenTelemetry GenAI spans."""
```

Extends `langchain_core.callbacks.BaseCallbackHandler` and implements these methods:

### Callback Methods Implemented

**Agent/Chain Events**:
```python
def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)
def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs)
def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs)
```

**Model Invocations**:
```python
def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)
def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)
def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs)
def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs)
```

**Tool Execution**:
```python
def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs)
def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs)
def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs)
```

**Agent Actions**:
```python
def on_agent_action(self, action, *, run_id, parent_run_id=None, **kwargs)
def on_agent_finish(self, finish, *, run_id, parent_run_id=None, **kwargs)
```

**Retriever Operations**:
```python
def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, metadata=None, **kwargs)
def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kwargs)
def on_retriever_error(self, error, *, run_id, parent_run_id=None, **kwargs)
```

### Usage Pattern

```python
tracer = AzureAIOpenTelemetryTracer(
    enable_content_recording=True,
    name="my-agent"
)

# Attach to graph config
app = workflow.compile(name="my-graph").with_config({
    "callbacks": [tracer]
})

# Or pass to invoke
result = app.invoke(input_data, config={"callbacks": [tracer]})
```

---

## 5. Multi-Agent & Chain Invocation with Traceparent Propagation

### Pattern 1: Explicit Parent-Child with `parent_run_id`

When LangChain invokes nested chains/agents, it passes `parent_run_id`:

```python
def on_chain_start(
    self,
    serialized: dict[str, Any],
    inputs: dict[str, Any],
    *,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    ...
):
    resolved_parent = self._resolve_parent_id(parent_run_id)
    self._start_span(
        run_id,
        span_name,
        operation="invoke_agent",
        parent_run_id=effective_parent_run_id,
        ...
    )
```

### Pattern 2: AsyncIO Context Propagation

**Location**: `inference_tracing.py`, lines 93-102

```python
_inherited_agent_context: ContextVar[Optional[Tuple[str, Any]]] = ContextVar(
    "_inherited_agent_context", default=None
)
```

**Explanation**:
- When `asyncio.create_task()` is used, child tasks inherit `ContextVar` values
- The tracer publishes agent span context in a `ContextVar` at span creation
- Orphaned `invoke_agent` spans automatically inherit parent context via this mechanism
- Eliminates need for explicit traceparent headers in multi-threaded dispatch

**Implementation in `_start_span` (lines 2413-2415)**:

```python
if operation == "invoke_agent":
    token = _inherited_agent_context.set((run_key, span.get_span_context()))
    span_record.stash["_inherited_ctx_token"] = token
```

**Context resolution in `_start_span` (lines 2356-2374)**:

```python
# Try inherited context from asyncio.create_task() before
# falling through to the ambient OTel span.
inherited = _inherited_agent_context.get(None)
if inherited is not None and operation == "invoke_agent":
    inherited_run_key, inherited_span_ctx = inherited
    # If the parent span is still alive, use it directly.
    if inherited_run_key in self._spans:
        parent_record = self._spans[inherited_run_key]
        resolved_parent_key = inherited_run_key
        parent_context = set_span_in_context(parent_record.span)
        parent_source = "contextvar"
    elif inherited_span_ctx is not None:
        # Parent already ended — link via its SpanContext.
        from opentelemetry.trace import NonRecordingSpan
        parent_context = set_span_in_context(
            NonRecordingSpan(inherited_span_ctx)
        )
        parent_source = "contextvar_detached"
```

### Pattern 3: LangGraph `goto` Command Handling

**Location**: `inference_tracing.py`, lines 2023-2030 (in `on_tool_end`)

```python
unwrapped, goto = _unwrap_command_like(output)
if goto:
    record.span.set_attribute("metadata.langgraph.goto", goto)
    record.attributes["metadata.langgraph.goto"] = goto
    parent_for_goto = self._nearest_invoke_agent_parent(record)
    self._push_goto_parent(record.stash.get("thread_id"), parent_for_goto)
```

**How it works**:
1. When a LangGraph node returns a `Command` with a `goto` target
2. The tracer records the goto destination
3. Pushes the parent onto a goto stack per thread
4. When the goto'd node is invoked, it uses this stack to find the correct parent

**Test Example (test_agent_tracing_e2e.py, lines 302-330)**:

```python
def test_goto_command_parents_agent_spans() -> None:
    tracer = RecordingTracer(enable_content_recording=True, name="goto-parent")

    def agent_a(state: MessagesState) -> Command:
        return Command(
            update={"messages": [AIMessage(content="routing to agent b")]},
            goto="agent_b",
        )

    def agent_b(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        return {"messages": [AIMessage(content="done")]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent_a", agent_a, metadata={"otel_agent_span": True})
    workflow.add_node("agent_b", agent_b, metadata={"otel_agent_span": True})
    ...
    
    # Assertion: agent_b's parent is agent_a due to goto tracking
    assert agent_spans["agent_b"].parent_run_id == agent_spans["agent_a"].run_id
```

### Pattern 4: Thread-Based Context for Concurrent Execution

**Location**: `inference_tracing.py`, lines 2323-2354

The tracer maintains per-thread agent stacks to handle concurrent agents:

```python
self._agent_stack_by_thread: Dict[str, List[str]] = {}

# In _start_span:
stack = self._agent_stack_by_thread.get(thread_str) if thread_str else None
if operation == "invoke_agent":
    stack = self._agent_stack_by_thread.setdefault(str(thread_key), [])
    stack.append(run_key)
```

This allows multiple agents running concurrently in the same thread to maintain proper hierarchy.

### Real-World Multi-Agent Example

**From test_agent_tracing_e2e.py (lines 386-465)**:

```python
async def test_langgraph_agent_loop_records_spans():
    """Multi-node graph with tool loop."""
    tracer = RecordingTracer(enable_content_recording=True, name="calculator-agent")

    @tool
    def add_numbers(a: int, b: int) -> int:
        return a + b

    model = _get_openai_model()

    def should_continue(state: MessagesState) -> str:
        return "continue" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    async def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        prompt = [SystemMessage(...), *state["messages"]]
        response = await model.bind_tools([add_numbers]).ainvoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode([add_numbers]))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END},
    )
    workflow.add_edge("action", "agent")
    
    # Span hierarchy created:
    # invoke_agent calculator-agent (root)
    #  ├─ chat gpt-4.1 (model invocation)
    #  └─ execute_tool add_numbers (tool execution)
```

---

## 6. W3C Trace Context & OpenTelemetry Integration

### OpenTelemetry Imports

**Location**: `inference_tracing.py`, lines 59-78

```python
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import attach, detach
    from opentelemetry.propagate import extract
    from opentelemetry.semconv.schemas import Schemas
    from opentelemetry.trace import (
        Span, SpanKind, Status, StatusCode, 
        get_current_span, set_span_in_context,
    )
except ImportError as exc:
    raise ImportError(
        "Azure OpenTelemetry tracing requires 'azure-monitor-opentelemetry' "
        "and 'opentelemetry-sdk'. Install them via:\n"
        "    pip install azure-monitor-opentelemetry opentelemetry-sdk"
    ) from exc
```

### Semantic Conventions

**Location**: `_semantic_conventions_gen_ai.py`

Constants following OpenTelemetry GenAI semantic conventions:

```python
GEN_AI_MESSAGE_ID = "gen_ai.message.id"
GEN_AI_MESSAGE_STATUS = "gen_ai.message.status"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
```

### Span Attributes Class

**Location**: `inference_tracing.py`, lines 105-155

```python
class Attrs:
    """Semantic convention attribute names used throughout the tracer."""
    
    PROVIDER_NAME = "gen_ai.provider.name"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"
    CONVERSATION_ID = "gen_ai.conversation.id"
    RETRIEVER_RESULTS = "gen_ai.retriever.results"
    RETRIEVER_QUERY = "gen_ai.retriever.query"
    # OpenAI-specific
    OPENAI_REQUEST_SERVICE_TIER = "openai.request.service_tier"
    OPENAI_RESPONSE_SERVICE_TIER = "openai.response.service_tier"
```

### Span Creation

**Location**: `inference_tracing.py`, lines 2395-2400

```python
span = self._tracer.start_span(
    name=name,
    context=parent_context,  # W3C context propagated here
    kind=kind,               # SpanKind.CLIENT, SERVER, INTERNAL, etc.
    attributes=attributes or {},
)
```

### Provider Detection

**Location**: `inference_tracing.py`, lines 914-974

Auto-detects provider from invocation params and serialization:

```python
def _infer_provider_name(
    serialized: Optional[dict[str, Any]],
    metadata: Optional[dict[str, Any]],
    invocation_params: Optional[dict[str, Any]],
) -> Optional[str]:
    # Returns: "azure.ai.openai", "openai", "aws.bedrock", "ollama", etc.
    # Inspects:
    # - metadata.ls_provider
    # - invocation_params.base_url
    # - serialized.kwargs
```

### OTEL Export Configuration

**Environment Variables Supported**:

```bash
# Azure Monitor
APPLICATION_INSIGHTS_CONNECTION_STRING=...

# OTLP Collector
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
```

**From docstring (lines 4-10)**:

> It supports simultaneous export to Azure Monitor (when a connection string is supplied) 
> and to any OTLP collector configured via environment variables (for example::
>
>     export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
>     export OTEL_EXPORTER_OTLP_PROTOCOL=grpc

---

## 7. Span Lifecycle Management

### Span Record Data Structure

**Location**: `inference_tracing.py`, lines 1082-1090

```python
@dataclass
class _SpanRecord:
    run_id: str
    span: Span
    operation: str
    parent_run_id: Optional[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    stash: Dict[str, Any] = field(default_factory=dict)  # Arbitrary data per span
```

### Span Creation Flow

1. **`_start_span`** (lines 2307-2442):
   - Resolves parent span context from multiple sources
   - Creates OpenTelemetry span with proper parent linkage
   - Records span in `self._spans` dict
   - Publishes agent context via ContextVar if applicable
   - Maintains thread-local agent stack

2. **Span Operation Types**:
   - `invoke_agent`: Agent/chain execution
   - `chat`: Chat model invocation
   - `text_completion`: LLM text completion
   - `execute_tool`: Tool or retriever execution

### Span Closure Flow

1. **`_end_span`** (lines 2444-2496):
   - Pops span from `self._spans` dict
   - Sets status (OK or ERROR)
   - Pops from agent stack if applicable
   - Resets ContextVar token to prevent context leakage
   - Calls `span.end()` to finalize with OTEL backend

### Example: Full Agent Span Lifecycle

**on_chain_start** creates span:
```python
def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, ...):
    trace_headers = _extract_trace_headers(metadata) or ...
    with self.use_propagated_context(headers=trace_headers):
        self._start_span(
            run_id,
            span_name=f"invoke_agent {agent_name}",
            operation="invoke_agent",
            kind=SpanKind.CLIENT,
            parent_run_id=effective_parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )
```

**on_chat_model_start** creates child span:
```python
def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, ...):
    self._handle_model_start(...)
    # Parent run_id will resolve to previous invoke_agent span
    self._start_span(
        run_id,
        span_name=f"chat {model_name}",
        operation="chat",
        kind=SpanKind.CLIENT,
        parent_run_id=parent_run_id,  # Points to agent span
        attributes=attributes,
    )
```

**on_llm_end** accumulates usage and closes span:
```python
def on_llm_end(self, response, *, run_id, parent_run_id=None, ...):
    record = self._spans.get(str(run_id))
    if record:
        # Extract and set token usage
        (input_tokens, output_tokens, total_tokens) = _extract_usage_tokens(...)
        record.span.set_attribute(Attrs.USAGE_INPUT_TOKENS, input_tokens)
        record.span.set_attribute(Attrs.USAGE_OUTPUT_TOKENS, output_tokens)
        # Propagate to parent agent span
        self._accumulate_usage_to_agent_spans(...)
        record.span.set_status(Status(StatusCode.OK))
    self._end_span(run_id)
```

**on_chain_end** closes agent span:
```python
def on_chain_end(self, outputs, *, run_id, parent_run_id=None, ...):
    record = self._spans.get(str(run_id))
    if record:
        # Record output messages
        formatted_messages, _ = self._prepare_messages_fn(...)
        record.span.set_attribute(Attrs.OUTPUT_MESSAGES, formatted_messages)
        # Propagate usage totals up
        self._propagate_agent_usage_totals(record)
        record.span.set_status(Status(StatusCode.OK))
    self._end_span(run_id)
```

---

## 8. Key Features & Advanced Patterns

### Content Redaction

**Location**: `inference_tracing.py`, lines 165-204

When `enable_content_recording=False`:
- Message content → `"[redacted]"`
- Tool results → `"[redacted]"`
- Tool arguments → `"[redacted]"`
- Numeric parameters preserved (temperature, tokens, etc.)

### Message Extraction with Path Overrides

**Location**: `inference_tracing.py`, lines 430-478

Supports multiple message key strategies:

```python
# Default: looks for "messages" key
message_keys=("messages",)

# Custom: look for different keys
message_keys=("chat_history", "conversation")

# Nested paths: dot-notation access
message_paths=("payload.messages", "state.history")

# Per-node override
metadata={
    "otel_messages_key": "chat_history",
    "otel_messages_path": "payload.nested.messages"
}
```

### Provider & Model Inference

Automatically detects:
- **Provider**: Azure OpenAI, OpenAI, GitHub Models, Ollama, AWS Bedrock
- **Model**: Extracts from invocation params or serialized config
- **Server**: Hostname/port from base_url or endpoint

### Token Usage Aggregation

**Location**: `inference_tracing.py`, lines 1322-1433

Accumulates token usage from child spans (models, tools) up to parent agent span:

```
invoke_agent root
├─ chat model (1000 input, 200 output)
├─ execute_tool retrieval (50 input, 100 output)
└─ chat model (1000 input, 150 output)

→ Agent span attributes:
  - gen_ai.usage.input_tokens = 2050
  - gen_ai.usage.output_tokens = 450
```

### Conversation ID Propagation

Thread ID is propagated to all child spans in the conversation:

```python
thread_identifier = metadata.get("thread_id") or metadata.get("session_id")
if thread_key:
    attributes[Attrs.CONVERSATION_ID] = thread_key
    # Propagate to all children
    self._update_parent_attribute(resolved_parent, Attrs.CONVERSATION_ID, thread_key)
```

### Node-Level Filtering

Control which LangGraph nodes get traced:

```python
metadata={
    "otel_agent_span": True,       # Force this node to create a span
    "otel_agent_span_allowed": ["llm_start"],  # Only allow certain callbacks
    "otel_trace": True,            # Explicit trace flag
}

# Or globally:
trace_all_langgraph_nodes=True     # Every node gets a span
ignore_start_node=True             # Skip __start__ node
```

---

## 9. Sample Usage

### Basic LangGraph Integration

**From samples/tracer_compat_sample.py**:

```python
from dataclasses import dataclass
from langgraph.graph import END, START, StateGraph
from langchain_azure_ai.callbacks.tracers.inference_tracing import AzureAIOpenTelemetryTracer

@dataclass
class State:
    chat_history: list[Any]
    final: str | None = None

async def main():
    tracer = AzureAIOpenTelemetryTracer(
        message_keys=("messages",),
        message_paths=("chat_history",),  # Messages live under chat_history
        trace_all_langgraph_nodes=True,
    )

    graph = (
        StateGraph(State)
        .add_node(
            "analyze",
            analyze,
            metadata={
                "otel_trace": True,
                "otel_messages_key": "chat_history",
                "langgraph_node": "analyze",
            },
        )
        .add_node(
            "review",
            review,
            metadata={
                "otel_trace": True,
                "otel_messages_key": "chat_history",
                "langgraph_node": "review",
            },
        )
        .add_edge(START, "analyze")
        .add_edge("analyze", "review")
        .add_edge("review", END)
        .compile(name="tracer-compat-graph")
        .with_config({"callbacks": [tracer]})
    )

    result = await graph.ainvoke(State(chat_history=[{"role": "user", "content": "hi"}]))
```

### With Azure Monitor

```python
tracer = AzureAIOpenTelemetryTracer(
    connection_string="InstrumentationKey=...;IngestionEndpoint=...",
    enable_content_recording=True,
    agent_id="my-agent-v1",
)
```

### With Project Resolution

```python
tracer = AzureAIOpenTelemetryTracer(
    project_endpoint="https://my-project.region.api.azureml.ms",
    # Uses DefaultAzureCredential to resolve telemetry connection
)
```

### Propagating Upstream Trace

```python
# From HTTP request
@app.post("/chat")
async def chat(request: Request, user_input: str):
    tracer = AzureAIOpenTelemetryTracer()
    
    # Extract trace headers from incoming HTTP request
    headers = {
        "traceparent": request.headers.get("traceparent"),
        "tracestate": request.headers.get("tracestate"),
    }
    
    # LangGraph spans will be children of HTTP span
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={
            "callbacks": [tracer],
            "metadata": {"traceparent_headers": headers}
        }
    )
```

---

## 10. Test Coverage

### Integration Test Scenarios

| Test | Scenario | Key Validation |
|------|----------|-----------------|
| `test_basic_agent_tracing_records_spans` | Single node agent | Agent span creation, conversation IDs |
| `test_agent_with_tool_records_tool_span` | Agent + tool execution | Tool span parentage, metadata |
| `test_langgraph_agent_loop_records_spans` | Multi-node with loops | Span hierarchy, usage propagation |
| `test_goto_command_parents_agent_spans` | LangGraph goto commands | Correct parent linkage for conditional flows |
| `test_trace_all_nodes_records_unlabeled_graph` | Unlabeled graph tracing | Global `trace_all_langgraph_nodes` flag |
| `test_metadata_message_path_records_wrapped_state` | Custom message paths | Nested dataclass extraction |
| `test_static_retriever_records_results` | Retriever operations | Query and document recording |
| `test_multi_turn_conversation_with_thread_id` | Conversation threading | Conversation ID persistence |

### Unit Test Coverage

- Message formatting and redaction
- Token usage extraction from various vendor formats
- Header extraction and context propagation
- Provider/model inference
- Bedrock/AWS-specific metric handling

---

## 11. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  HTTP Ingress / Orchestrator                 │
│                  (W3C traceparent header)                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ extract headers
                  ▼
┌─────────────────────────────────────────────────────────────┐
│          AzureAIOpenTelemetryTracer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ use_propagated_context(headers)                      │   │
│  │  └─ extract(headers) → OpenTelemetry context        │   │
│  │     └─ attach(ctx) before span creation             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ on_chain_start → _start_span                        │    │
│  │  ├─ Resolve parent from:                            │    │
│  │  │  1. Explicit parent_run_id                       │    │
│  │  │  2. Agent stack per thread                       │    │
│  │  │  3. Inherited ContextVar                         │    │
│  │  │  4. Ambient OTel span (current_span)             │    │
│  │  └─ Create span with parent context                 │    │
│  │     └─ Publish ContextVar for asyncio inheritance   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ on_chat_model_start → _start_span (child)           │    │
│  │  └─ Uses parent_run_id from callback                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ on_llm_end → Aggregate usage → on_chain_end         │    │
│  │  └─ Propagate tokens up to agent span               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ _end_span                                           │    │
│  │  └─ span.end() → Export to Azure Monitor / OTLP     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│            OpenTelemetry SDK                                 │
│  ┌───────────────────┐  ┌──────────────────────────────┐   │
│  │ Azure Monitor     │  │ OTLP Collector               │   │
│  │ Exporter          │  │ (HTTP/gRPC)                  │   │
│  └───────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Key Takeaways

### Design Principles

1. **Multi-Source Parent Resolution**: Handles explicit parents, context vars, thread stacks, and ambient spans
2. **Graceful Fallbacks**: Works with or without traceparent headers, with or without Azure Monitor
3. **Thread-Safe**: Uses locks for span dict access, ContextVars for async context
4. **Content Redaction**: Optional content recording with smart defaults for security
5. **Semantic Conventions**: Emits spec-compliant GenAI attributes for standardization
6. **Zero Configuration**: Works with reasonable defaults; supports custom providers/models

### Traceparent Propagation Strategy

1. **Extract W3C headers** from inputs/metadata/kwargs via `_extract_trace_headers()`
2. **Deserialize context** with OpenTelemetry's `extract()` function
3. **Temporarily attach** context before span creation with `use_propagated_context()`
4. **Create spans** with parent context, creating proper trace hierarchy
5. **Automatic asyncio propagation** via ContextVar for worker task inheritance

### No explicit passing required!

The tracer automatically:
- Detects parent from callback parent_run_id
- Inherits from asyncio task context (ContextVar)
- Falls back to ambient OpenTelemetry span
- Supports LangGraph goto command reparenting
- Maintains thread-local agent stacks for concurrent execution

---

**End of Analysis**
