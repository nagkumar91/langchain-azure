"""LangGraph “time travel DJ” sample instrumented with debug tracing.

Run with:

    /Users/nagkumar/Documents/msft.nosync/python-ai-agent-frameworks-demos/venv/bin/python \
        time_travel_debug_sample.py

Before running, edit `.env` in this directory with the credentials that match
the provider selected via `API_HOST`.
"""

from __future__ import annotations

import os
from pathlib import Path

import azure.identity
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_azure_ai.callbacks.tracers import (
    AzureAIOpenTelemetryTracer,
    DebuggingCallbackHandler,
)

LOG_PATH = Path(__file__).with_suffix(".log")
ENV_PATH = Path(__file__).with_name(".env")


def _load_environment() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def _build_model() -> ChatOpenAI:
    api_host = os.getenv("API_HOST", "github").lower()

    if api_host == "azure":
        token_provider = azure.identity.get_bearer_token_provider(
            azure.identity.DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureChatOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_ad_token_provider=token_provider,
        )

    if api_host == "github":
        return ChatOpenAI(
            model=os.getenv("GITHUB_MODEL", "gpt-4o"),
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_TOKEN"),
        )

    if api_host == "ollama":
        return ChatOpenAI(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
            api_key="none",
        )

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


@tool
def play_song_on_spotify(song: str) -> str:
    """Play a song on Spotify."""
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str) -> str:
    """Play a song on Apple Music."""
    return f"Successfully played {song} on Apple Music!"


def build_graph(model: ChatOpenAI) -> StateGraph:
    tools = [play_song_on_apple, play_song_on_spotify]
    tool_node = ToolNode(tools)
    bound_model = model.bind_tools(tools, parallel_tool_calls=False)

    def should_continue(state: MessagesState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        return "end" if not last_message.tool_calls else "continue"

    def call_model(state: MessagesState) -> dict[str, list]:
        messages = state["messages"]
        response = bound_model.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")
    return workflow


def main() -> None:
    _load_environment()

    azure_tracer = AzureAIOpenTelemetryTracer(
        connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
        enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
        name="Music Player Agent",
    )
    debug_callback = DebuggingCallbackHandler(
        log_path=LOG_PATH,
        name="TimeTravelDebugCallback",
    )

    model = _build_model()
    graph = build_graph(model)
    app = graph.compile(checkpointer=MemorySaver())

    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [azure_tracer, debug_callback],
    }
    input_message = HumanMessage(content="Can you play Taylor Swift's most popular song?")

    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
