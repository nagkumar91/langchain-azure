"""Single-agent weekend planner sample instrumented with debug tracing."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from rich import print
from rich.logging import RichHandler

from langchain_azure_ai.callbacks.tracers import (
    AzureAIOpenTelemetryTracer,
    DebuggingAgentMiddleware,
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


logger = logging.getLogger("weekend_planner")


@tool
def get_weather(city: str, date: str) -> dict:
    """Returns weather data for a given city and date."""
    logger.info("Getting weather for %s on %s", city, date)
    if random.random() < 0.05:
        return {"temperature": 72, "description": "Sunny"}
    return {"temperature": 60, "description": "Rainy"}


@tool
def get_activities(city: str, date: str) -> list:
    """Returns a list of activities for a given city and date."""
    logger.info("Getting activities for %s on %s", city, date)
    return [
        {"name": "Hiking", "location": city},
        {"name": "Beach", "location": city},
        {"name": "Museum", "location": city},
    ]


@tool
def get_current_date() -> str:
    """Gets the current date from the system."""
    logger.info("Getting current date")
    return datetime.now().strftime("%Y-%m-%d")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    logger.setLevel(logging.INFO)

    _load_environment()

    azure_tracer = AzureAIOpenTelemetryTracer(
        connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
        enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
        name="Weekend Planner Agent",
    )
    debug_callback = DebuggingCallbackHandler(
        log_path=LOG_PATH,
        name="WeekendPlannerCallback",
    )
    debug_middleware = DebuggingAgentMiddleware(
        log_path=LOG_PATH,
        name="WeekendPlannerMiddleware",
        include_runtime_snapshot=True,
    )

    model = _build_model()
    agent = create_agent(
        model=model,
        system_prompt=(
            "You help users plan their weekends and choose the best activities "
            "for the given weather. If an activity would be unpleasant in the "
            "weather, don't suggest it. Include the date of the weekend in your response."
        ),
        tools=[get_weather, get_activities, get_current_date],
        middleware=[debug_middleware],
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "hi what can I do this weekend in San Francisco?"}]},
        config={"callbacks": [azure_tracer, debug_callback]},
    )
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    main()
