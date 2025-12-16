from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

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


@pytest.mark.asyncio
@pytest.mark.block_network()
async def test_trace_all_nodes_records_unlabeled_graph() -> None:
    """Tracer should emit spans for every node when trace_all_langgraph_nodes=True."""
    tracer = RecordingTracer(
        enable_content_recording=True,
        name="trace-all",
        trace_all_langgraph_nodes=True,
        message_paths=("chat_history",),
    )

    @dataclass
    class State:
        chat_history: List[Any]

    async def gather(state: State, runtime: Any | None = None) -> Dict[str, Any]:
        return {
            "chat_history": state.chat_history
            + [{"role": "assistant", "content": "gathered"}]
        }

    async def summarize(state: State, runtime: Any | None = None) -> Dict[str, Any]:
        return {
            "chat_history": state.chat_history
            + [{"role": "assistant", "content": "summarized"}]
        }

    graph = (
        StateGraph(State)
        .add_node("gather", gather)
        .add_node("summarize", summarize)
        .add_edge(START, "gather")
        .add_edge("gather", "summarize")
        .add_edge("summarize", END)
        .compile(name="trace-all-graph")
        .with_config({"callbacks": [tracer]})
    )

    await graph.ainvoke(State(chat_history=[{"role": "user", "content": "hi"}]))

    span_names = [span.name for span in tracer.completed_spans]
    assert any("gather" in name for name in span_names)
    assert any("summarize" in name for name in span_names)


@pytest.mark.asyncio
@pytest.mark.block_network()
async def test_metadata_message_path_records_wrapped_state() -> None:
    """Node-level otel_messages_path should extract nested dataclass state."""
    tracer = RecordingTracer(enable_content_recording=True, name="wrapped-state")

    @dataclass
    class WrappedState:
        payload: Dict[str, Any]

    async def enrich(state: WrappedState, runtime: Any | None = None) -> Dict[str, Any]:
        history = state.payload["messages"]
        return {
            "payload": {
                "messages": history + [{"role": "assistant", "content": "wrapped"}]
            }
        }

    graph = (
        StateGraph(WrappedState)
        .add_node(
            "enrich",
            enrich,
            metadata={
                "otel_trace": True,
                "otel_messages_path": "payload.messages",
                "langgraph_node": "enrich",
            },
        )
        .add_edge(START, "enrich")
        .add_edge("enrich", END)
        .compile(name="wrapped-state-graph")
        .with_config({"callbacks": [tracer]})
    )

    await graph.ainvoke(WrappedState(payload={"messages": [{"role": "user", "content": "hi"}]}))

    span = next(span for span in tracer.completed_spans if "enrich" in span.name)
    input_messages = json.loads(span.attributes[Attrs.INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == "hi"
    assert span.attributes.get("metadata.langgraph_node") == "enrich"


class _StaticRetriever(BaseRetriever):
    docs: List[Document]
    tags: List[str] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: RunnableConfig | None = None,
    ) -> List[Document]:
        return self.docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: RunnableConfig | None = None,
    ) -> List[Document]:
        return self.docs


@pytest.mark.block_network()
def test_static_retriever_records_results() -> None:
    """Ensure retriever spans record queries and documents."""
    tracer = RecordingTracer(enable_content_recording=True, name="retriever")
    docs = [
        Document(page_content="alpha", metadata={"chunk": 1}),
        Document(page_content="beta", metadata={"chunk": 2}),
    ]
    retriever = _StaticRetriever(docs=docs)

    retriever.invoke("alpha", config={"callbacks": [tracer]})

    span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "execute_tool" and span.attributes.get(Attrs.TOOL_TYPE) == "retriever"
    )
    assert span.attributes.get(Attrs.RETRIEVER_QUERY) == "alpha"
    recorded = json.loads(span.attributes[Attrs.RETRIEVER_RESULTS])
    assert recorded[0]["metadata"]["chunk"] == 1


@tool
def exploding_tool(text: str) -> str:
    """Always raises an exception."""
    raise ValueError("boom")


@pytest.mark.block_network()
def test_tool_error_span_records_status() -> None:
    """ToolNode errors should emit execute_tool spans with error metadata."""
    tracer = RecordingTracer(enable_content_recording=True, name="error-tool")
    tool_node = ToolNode([exploding_tool])

    graph = (
        StateGraph(MessagesState)
        .add_node(
            "tools",
            tool_node,
            metadata={"otel_trace": True, "langgraph_node": "tools"},
        )
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile(name="error-tool-graph")
        .with_config({"callbacks": [tracer]})
    )

    ai_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(id="call-1", name="exploding_tool", args={"text": "hi"})
        ],
    )

    with pytest.raises(ValueError):
        graph.invoke({"messages": [ai_message]})

    span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "execute_tool" and "exploding_tool" in span.name
    )
    assert span.attributes.get(Attrs.TOOL_CALL_RESULT) is None


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
