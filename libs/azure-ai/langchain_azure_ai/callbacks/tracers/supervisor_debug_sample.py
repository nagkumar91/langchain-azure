"""Multi-agent supervisor sample instrumented with Azure OTEL tracer."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from rich import print
from rich.logging import RichHandler

from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

from opentelemetry import trace as otel_trace

ENV_PATH = Path(__file__).with_name(".env")
LOGGER = logging.getLogger(__name__)


def _load_environment() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=True)


def _configure_otlp_exporter() -> None:
    endpoint = (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if not endpoint:
        return
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()
    try:
        provider = otel_trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            LOGGER.warning("Tracer provider does not support span processors")
            return
        if protocol in {"grpc", "grpc/protobuf"}:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        elif protocol in {"http", "http/protobuf"}:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            LOGGER.warning("Unsupported OTLP protocol '%s'", protocol)
            return
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to configure OTLP exporter: %s", exc)


def _build_base_model() -> ChatOpenAI:
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


logger = logging.getLogger("lang_triage")


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


@tool
def find_recipes(query: str) -> list[dict]:
    """Returns recipes based on a query."""
    logger.info("Finding recipes for %s", query)
    lower = query.lower()
    if "pasta" in lower:
        return [
            {
                "title": "Pasta Primavera",
                "ingredients": ["pasta", "vegetables", "olive oil"],
                "steps": ["Cook pasta.", "Sauté vegetables."],
            }
        ]
    if "tofu" in lower:
        return [
            {
                "title": "Tofu Stir Fry",
                "ingredients": ["tofu", "soy sauce", "vegetables"],
                "steps": ["Cube tofu.", "Stir fry veggies."],
            }
        ]
    return [
        {
            "title": "Grilled Cheese Sandwich",
            "ingredients": ["bread", "cheese", "butter"],
            "steps": [
                "Butter bread.",
                "Place cheese between slices.",
                "Grill until golden brown.",
            ],
        }
    ]


@tool
def check_fridge() -> list[str]:
    """Returns a list of ingredients currently in the fridge."""
    logger.info("Checking fridge for current ingredients")
    if random.random() < 0.5:
        return ["pasta", "tomato sauce", "bell peppers", "olive oil"]
    return ["tofu", "soy sauce", "broccoli", "carrots"]


def _configure_tracer(name: str) -> AzureAIOpenTelemetryTracer:
    tracer = AzureAIOpenTelemetryTracer(
        connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
        enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
        name=name,
    )
    return tracer


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    logger.setLevel(logging.INFO)

    _load_environment()

    base_model = _build_base_model()

    supervisor_tracer = _configure_tracer("Supervisor Agent")
    activity_tracer = _configure_tracer("Weekend Activity Planner")
    meal_tracer = _configure_tracer("Meal Recipe Planner")
    _configure_otlp_exporter()

    weekend_agent = create_agent(
        model=base_model,
        system_prompt=(
            "You help users plan their weekends and choose the best activities for the "
            "given weather. If an activity would be unpleasant in the weather, don't "
            "suggest it. Include the date of the weekend in your response."
        ),
        tools=[get_weather, get_activities, get_current_date],
    )

    @tool
    def plan_weekend(query: str) -> str:
        """Delegate weekend planning to the activity agent and return its reply."""
        logger.info("Tool: plan_weekend invoked")
        response = weekend_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"callbacks": [activity_tracer]},
        )
        return response["messages"][-1].content

    meal_agent = create_agent(
        model=base_model,
        system_prompt=(
            "You help users plan meals and choose the best recipes. Include the ingredients "
            "and cooking instructions in your response. Indicate what the user needs to buy "
            "from the store when their fridge is missing ingredients."
        ),
        tools=[find_recipes, check_fridge],
    )

    @tool
    def plan_meal(query: str) -> str:
        """Delegate meal planning to the recipe agent and return its reply."""
        logger.info("Tool: plan_meal invoked")
        response = meal_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"callbacks": [meal_tracer]},
        )
        return response["messages"][-1].content

    supervisor_agent = create_agent(
        model=base_model,
        system_prompt=(
            "You are a supervisor, managing an activity planning agent and recipe planning "
            "agent. Assign work to them as needed in order to answer the user's question."
        ),
        tools=[plan_weekend, plan_meal],
    )

    response = supervisor_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "my kids want pasta for dinner, give me weekend plans",
                }
            ]
        },
        config={"callbacks": [supervisor_tracer]},
    )
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    main()
