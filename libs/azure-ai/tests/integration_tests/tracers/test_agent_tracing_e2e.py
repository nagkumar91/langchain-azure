from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    Attrs,
    AzureAIOpenTelemetryTracer,
)


@dataclass
class _RecordedSpan:
    name: str
    operation: str
    run_id: str
    parent_run_id: str | None
    attributes: Dict[str, Any]


class RecordingTracer(AzureAIOpenTelemetryTracer):
    """Tracer that records completed spans for assertions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.completed_spans: List[_RecordedSpan] = []
        self._span_names: Dict[str, str] = {}

    def _start_span(  # type: ignore[override]
        self,
        run_id: UUID,
        name: str,
        *,
        operation: str,
        kind: Any,
        parent_run_id: UUID | None,
        attributes: Dict[str, Any] | None = None,
        thread_key: str | None = None,
    ) -> None:
        super()._start_span(
            run_id,
            name,
            operation=operation,
            kind=kind,
            parent_run_id=parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )
        self._span_names[str(run_id)] = name

    def _end_span(  # type: ignore[override]
        self,
        run_id: UUID,
        *,
        status: Any = None,
        error: Any = None,
    ) -> None:
        run_key = str(run_id)
        record = self._spans.get(run_key)
        span_name = self._span_names.pop(run_key, None)
        operation = record.operation if record else ""
        run_id_str = record.run_id if record else run_key
        parent_run = record.parent_run_id if record else None
        attributes = dict(record.attributes) if record else {}
        super()._end_span(run_id, status=status, error=error)
        if span_name is not None:
            self.completed_spans.append(
                _RecordedSpan(
                    name=span_name,
                    operation=operation,
                    run_id=run_id_str,
                    parent_run_id=parent_run,
                    attributes=attributes,
                )
            )


class ToolAwareFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Fake chat model that allows binding tools by returning itself."""

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> FakeMessagesListChatModel:
        self._bound_tools = tools  # type: ignore[attr-defined]
        return self


@pytest.mark.block_network()
def test_basic_agent_tracing_records_spans() -> None:
    """Ensure a simple agent invocation produces agent and LLM spans."""
    tracer = RecordingTracer(enable_content_recording=True, name="informational-agent")
    responses = cast(List[BaseMessage], [AIMessage(content="It's always sunny!")])
    model = FakeMessagesListChatModel(responses=responses)
    agent: Any = create_agent(
        model=model,
        system_prompt="You are an informational agent. Answer questions cheerfully.",
        tools=[],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather today?"}]},
        config={"callbacks": [tracer]},
    )

    assert result["messages"][-1].content == "It's always sunny!"
    span_names = [span.name for span in tracer.completed_spans]
    assert any(name.startswith("invoke_agent") for name in span_names)
    chat_span = next(
        span for span in tracer.completed_spans if span.operation == "chat"
    )
    assert chat_span.attributes.get(Attrs.OPERATION_NAME) == "chat"
    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "LangGraph"
    assert chat_span.parent_run_id == root_span.run_id


@pytest.mark.block_network()
def test_agent_with_tool_records_tool_span() -> None:
    """Ensure tool execution is traced with arguments and results."""
    tracer = RecordingTracer(enable_content_recording=True, name="weather-agent")

    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "id": "call_weather",
                "args": {"city": "San Francisco"},
            }
        ],
    )
    final_reply = AIMessage(content="It's rainy and 60°F in San Francisco.")
    responses = cast(List[BaseMessage], [tool_call_message, final_reply])
    model = ToolAwareFakeMessagesListChatModel(responses=responses)

    @tool
    def get_weather(city: str) -> dict[str, Any]:
        """Return mock weather information for a city."""
        return {"temperature": 60, "description": "Rainy", "city": city}

    agent: Any = create_agent(
        model=model,
        system_prompt="You are an informational agent. Answer questions cheerfully.",
        tools=[get_weather],
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "weather in San Francisco?"}]},
        config={"callbacks": [tracer]},
    )

    assert response["messages"][-1].content == "It's rainy and 60°F in San Francisco."

    tool_span = next(
        span
        for span in tracer.completed_spans
        if span.name.startswith("execute_tool get_weather")
    )
    assert tool_span.operation == "execute_tool"
    assert tool_span.attributes.get(Attrs.TOOL_CALL_ARGUMENTS)

    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "LangGraph"
    span_index = {span.run_id: span for span in tracer.completed_spans}
    parent_span = span_index.get(tool_span.parent_run_id or "")
    assert parent_span is not None
    assert parent_span.operation in {"invoke_agent", "chat"}


@pytest.mark.block_network()
def test_langgraph_agent_loop_records_spans() -> None:
    """Ensure LangGraph agent/tool loop traces correctly."""
    tracer = RecordingTracer(enable_content_recording=True, name="music-agent")

    @tool
    def play_song_on_spotify(song: str) -> str:
        """Play a song on Spotify."""
        return f"Successfully played {song} on Spotify!"

    @tool
    def play_song_on_apple(song: str) -> str:
        """Play a song on Apple Music."""
        return f"Successfully played {song} on Apple Music!"

    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "play_song_on_apple",
                "id": "call_music",
                "args": {"song": "Taylor Swift's most popular song"},
            }
        ],
    )
    final_reply = AIMessage(content="Playing Taylor Swift's most popular song!")
    responses = cast(List[BaseMessage], [tool_call_message, final_reply])
    model = ToolAwareFakeMessagesListChatModel(responses=responses)
    model.bind_tools(
        [play_song_on_apple, play_song_on_spotify],
        parallel_tool_calls=False,
    )

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        return "continue" if getattr(last_message, "tool_calls", None) else "end"

    def call_model(state: MessagesState) -> Dict[str, List[AIMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode([play_song_on_apple, play_song_on_spotify]))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END},
    )
    workflow.add_edge("action", "agent")
    app = workflow.compile(checkpointer=MemorySaver())

    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "music-thread"}, "callbacks": [tracer]},
    )
    input_message = HumanMessage(
        content="Can you play Taylor Swift's most popular song?"
    )
    final_message: Optional[AIMessage] = None
    initial_state: MessagesState = {"messages": [input_message]}
    for event in app.stream(
        cast(Any, initial_state),
        config=config,
        stream_mode="values",
    ):
        final_message = cast(AIMessage, event["messages"][-1])

    assert final_message is not None
    assert "Taylor Swift" in final_message.content

    execute_span = next(
        span for span in tracer.completed_spans if span.operation == "execute_tool"
    )
    assert execute_span.attributes.get(Attrs.TOOL_NAME) == "play_song_on_apple"

    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "LangGraph"


@pytest.mark.block_network()
def test_agent_with_failing_tool_records_error_span() -> None:
    """Ensure tool failure is traced with error status."""
    tracer = RecordingTracer(enable_content_recording=True, name="error-agent")

    @tool
    def failing_tool(query: str) -> str:
        """A tool that always fails."""
        raise ValueError("Tool intentionally failed")

    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "failing_tool",
                "id": "call_fail",
                "args": {"query": "test"},
            }
        ],
    )
    final_reply = AIMessage(content="I encountered an error.")
    responses = cast(List[BaseMessage], [tool_call_message, final_reply])
    model = ToolAwareFakeMessagesListChatModel(responses=responses)

    agent: Any = create_agent(
        model=model,
        system_prompt="You are a helpful agent.",
        tools=[failing_tool],
    )

    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "Run the tool"}]},
            config={"callbacks": [tracer]},
        )
    except ValueError:
        pass  # Expected

    # There should be at least one span with execute_tool operation
    tool_spans = [
        span for span in tracer.completed_spans if span.operation == "execute_tool"
    ]
    assert len(tool_spans) > 0
    tool_span = tool_spans[0]
    assert tool_span.attributes.get(Attrs.TOOL_NAME) == "failing_tool"


@pytest.mark.block_network()
def test_multi_turn_conversation_with_thread_id() -> None:
    """Ensure multi-turn conversations maintain thread context."""
    tracer = RecordingTracer(enable_content_recording=True, name="multi-turn-agent")

    responses1 = cast(List[BaseMessage], [AIMessage(content="Hello! How can I help?")])
    responses2 = cast(List[BaseMessage], [AIMessage(content="The weather is sunny.")])

    model = FakeMessagesListChatModel(responses=responses1 + responses2)

    def call_model(state: MessagesState) -> Dict[str, List[AIMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    app = workflow.compile(checkpointer=MemorySaver())

    thread_id = "multi-turn-thread-123"
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": thread_id}, "callbacks": [tracer]},
    )

    # First turn
    input1: MessagesState = {"messages": [HumanMessage(content="Hi there!")]}
    result1 = app.invoke(cast(Any, input1), config=config)
    assert result1["messages"][-1].content == "Hello! How can I help?"

    # Second turn
    input2: MessagesState = {
        "messages": [
            *result1["messages"],
            HumanMessage(content="What's the weather?"),
        ]
    }
    result2 = app.invoke(cast(Any, input2), config=config)
    assert result2["messages"][-1].content == "The weather is sunny."

    # Verify spans were created for both turns
    root_spans = [
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    ]
    assert len(root_spans) >= 2
    # All root spans should have conversation ID attribute set
    for span in root_spans:
        assert span.attributes.get(Attrs.CONVERSATION_ID) == thread_id


@pytest.mark.block_network()
def test_agent_with_content_recording_disabled() -> None:
    """Ensure content is redacted when content recording is disabled."""
    tracer = RecordingTracer(enable_content_recording=False, name="redacted-agent")
    responses = cast(List[BaseMessage], [AIMessage(content="Secret response")])
    model = FakeMessagesListChatModel(responses=responses)
    agent: Any = create_agent(
        model=model,
        system_prompt="Secret system prompt",
        tools=[],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Secret user message"}]},
        config={"callbacks": [tracer]},
    )

    assert result["messages"][-1].content == "Secret response"

    chat_span = next(
        span for span in tracer.completed_spans if span.operation == "chat"
    )
    # Input messages should be redacted
    input_messages = chat_span.attributes.get(Attrs.INPUT_MESSAGES)
    if input_messages:
        assert "[redacted]" in input_messages
