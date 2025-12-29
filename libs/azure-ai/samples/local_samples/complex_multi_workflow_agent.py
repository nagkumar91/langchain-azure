"""Complex Multi-Workflow Agent Sample with GenAI Semantic Conventions.

This sample demonstrates:
1. **Routing workflow** - Classify requests and route to specialized handlers
2. **Orchestrator-worker pattern** - Parallel task execution with result synthesis
3. **Evaluator-optimizer loop** - Generate and iteratively improve responses
4. **Tool-calling agent (ReAct)** - Dynamic tool selection and execution
5. **Complex state management** - Nested dataclass state with custom message paths

The sample uses the static autolog() API to automatically trace all LangChain/LangGraph
executions without explicit callbacks. All spans follow the OpenTelemetry GenAI
semantic conventions:
  - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

Run with Python from repo root venv:
  .venv-py311/bin/python libs/azure-ai/samples/local_samples/complex_multi_workflow_agent.py

Environment variables:
  Azure OpenAI (Azure AI Foundry) for LLM calls:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_CHAT_COMPLETIONS_URL (preferred) OR
      AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT + AZURE_OPENAI_API_VERSION
  Public OpenAI for LLM calls:
    - OPENAI_API_KEY
  APPLICATIONINSIGHTS_CONNECTION_STRING for Azure Monitor (optional)
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT for OTLP export (optional)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Annotated
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from operator import add

from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer


# =============================================================================
# Tool Definitions - Multiple tools for different purposes
# =============================================================================


@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query to look up

    Returns:
        Search results as a string
    """
    # Simulated search results
    results = {
        "weather": "Current weather: Sunny, 72°F. Forecast: Clear skies expected.",
        "news": "Breaking: Major tech companies announce AI partnerships.",
        "travel": "Popular destinations: Paris, Tokyo, New York. Best time to visit varies.",
        "food": "Top restaurants: Le Bernardin (NYC), Eleven Madison Park, Noma.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': Found 10 relevant articles covering the topic."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2 * 3')

    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # Note: In production, use a safer parser
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_weather(location: str, units: str = "fahrenheit") -> str:
    """Get current weather for a location.

    Args:
        location: City or location name
        units: Temperature units - 'fahrenheit' or 'celsius'

    Returns:
        Current weather information
    """
    # Simulated weather data
    temps = {"fahrenheit": random.randint(60, 85), "celsius": random.randint(15, 30)}
    conditions = ["Sunny", "Partly Cloudy", "Overcast", "Light Rain"]
    return json.dumps({
        "location": location,
        "temperature": temps.get(units, temps["fahrenheit"]),
        "units": units,
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 80),
        "wind_speed": random.randint(5, 25),
    })


@tool
def query_database(table: str, filters: str = "") -> str:
    """Query a database table with optional filters.

    Args:
        table: Name of the database table to query
        filters: Optional filter conditions (e.g., 'status=active')

    Returns:
        Query results as JSON
    """
    # Simulated database query
    mock_data = {
        "users": [
            {"id": 1, "name": "Alice", "status": "active"},
            {"id": 2, "name": "Bob", "status": "inactive"},
            {"id": 3, "name": "Carol", "status": "active"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "total": 150.00, "status": "shipped"},
            {"id": 102, "user_id": 2, "total": 89.99, "status": "pending"},
        ],
        "products": [
            {"id": "A1", "name": "Widget", "price": 29.99, "stock": 100},
            {"id": "B2", "name": "Gadget", "price": 49.99, "stock": 50},
        ],
    }
    data = mock_data.get(table, [])
    if filters and data:
        key, value = filters.split("=") if "=" in filters else (None, None)
        if key:
            data = [row for row in data if str(row.get(key)) == value]
    return json.dumps({"table": table, "count": len(data), "rows": data})


@tool
def retrieve_knowledge(topic: str, max_results: int = 3) -> str:
    """Retrieve knowledge base articles about a topic.

    Args:
        topic: The topic to search for in the knowledge base
        max_results: Maximum number of results to return

    Returns:
        Relevant knowledge base articles
    """
    # Simulated knowledge base
    kb = {
        "billing": [
            "Invoice payments are due within 30 days.",
            "Accepted payment methods: Credit card, ACH, Wire transfer.",
            "Contact billing@example.com for payment issues.",
        ],
        "technical": [
            "API rate limit: 1000 requests per minute.",
            "Authentication uses OAuth 2.0 bearer tokens.",
            "SDK available for Python, JavaScript, and Java.",
        ],
        "policy": [
            "Returns accepted within 30 days of purchase.",
            "Refunds processed within 5-7 business days.",
            "Contact support for warranty claims.",
        ],
    }
    for key, articles in kb.items():
        if key in topic.lower():
            return json.dumps({
                "topic": topic,
                "articles": articles[:max_results],
                "total_found": len(articles),
            })
    return json.dumps({
        "topic": topic,
        "articles": ["No specific articles found. Please contact support."],
        "total_found": 0,
    })


ALL_TOOLS = [search_web, calculate, get_weather, query_database, retrieve_knowledge]


# =============================================================================
# State Definitions - Complex nested state with custom message paths
# =============================================================================


@dataclass
class WorkerResult:
    """Result from a worker agent."""
    worker_id: str
    task: str
    result: str
    confidence: float = 1.0


@dataclass
class EvaluationResult:
    """Result from the evaluator agent."""
    score: float
    feedback: str
    passed: bool


@dataclass
class ConversationThread:
    """Thread containing messages and metadata."""
    messages: List[BaseMessage]
    thread_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ComplexAgentState:
    """Complex state for multi-workflow agent.

    Attributes:
        conversation: The main conversation thread
        request_type: Classified type of the user request
        worker_results: Results from parallel worker agents
        evaluation: Result from evaluator agent
        iteration_count: Number of optimization iterations
        final_response: The final synthesized response
        metadata: Additional metadata for tracing
    """
    conversation: ConversationThread
    request_type: Optional[str] = None
    worker_results: List[WorkerResult] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    iteration_count: int = 0
    final_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# For tool-calling ReAct agent
class ReActState(TypedDict):
    """State for ReAct tool-calling agent."""
    messages: Annotated[List[BaseMessage], add]
    thread_id: str
    tool_calls_count: int


# =============================================================================
# LLM Factory - Support for Azure OpenAI and Public OpenAI
# =============================================================================


class OfflineLLM:
    """Offline fallback LLM for testing without API keys."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._bound_tools: List[Any] = []

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "OfflineLLM":
        self._bound_tools = tools
        return self

    def invoke(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        last_content = messages[-1].content if messages else "Request"

        # Simulate tool calls if tools are bound
        if self._bound_tools and "search" in last_content.lower():
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_web",
                    "id": f"call_{uuid4().hex[:8]}",
                    "args": {"query": "relevant information"},
                }]
            )

        return AIMessage(content=f"[{self.agent_name}] Processed: {last_content[:50]}...")


def create_llm(agent_name: str, temperature: float = 0.3) -> ChatOpenAI:
    """Create an LLM instance, preferring Azure OpenAI if configured."""

    def parse_azure_openai_chat_completions_url(url: str) -> tuple[str, str, Optional[str]]:
        parsed = urlparse(url)
        base_endpoint = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

        # Expected path shape:
        #   /openai/deployments/<deployment>/chat/completions
        path_parts = [p for p in parsed.path.split("/") if p]
        deployment: Optional[str] = None
        for idx, part in enumerate(path_parts):
            if part == "deployments" and idx + 1 < len(path_parts):
                deployment = path_parts[idx + 1]
                break

        api_version = parse_qs(parsed.query).get("api-version", [None])[0]

        if not base_endpoint or not deployment:
            raise ValueError(f"Invalid Azure OpenAI chat completions URL: {url}")

        return base_endpoint, deployment, api_version

    azure_completions_url = (
        os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_URL")
        or os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_ENDPOINT")
    )

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_BASE_URL")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")

    if azure_completions_url:
        azure_endpoint, deployment, api_version_from_url = parse_azure_openai_chat_completions_url(azure_completions_url)
        api_version = (
            api_version_from_url
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or "2024-10-21"
        )
    elif azure_endpoint and "/openai/deployments/" in azure_endpoint:
        azure_endpoint, deployment, api_version_from_url = parse_azure_openai_chat_completions_url(azure_endpoint)
        api_version = (
            api_version_from_url
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
            or "2024-10-21"
        )
    else:
        azure_endpoint = azure_endpoint.rstrip("/") if azure_endpoint else azure_endpoint
        deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or os.getenv("OPENAI_MODEL")
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-10-21"

    # Check if we have credentials
    has_azure = azure_endpoint and azure_key and deployment
    has_openai = os.getenv("OPENAI_API_KEY")

    if not has_azure and not has_openai:
        return OfflineLLM(agent_name)  # type: ignore

    if has_azure:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=temperature,
            max_tokens=500,
        )

    # Public OpenAI - temporarily clear Azure-specific env vars
    saved_vars = {}
    for var in ["OPENAI_API_VERSION", "OPENAI_BASE_URL"]:
        if var in os.environ:
            saved_vars[var] = os.environ.pop(var)
    try:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            temperature=temperature,
            max_tokens=500,
        )
    finally:
        os.environ.update(saved_vars)


# =============================================================================
# Node Metadata Helper - GenAI Semantic Conventions
# =============================================================================


def agent_metadata(
    node_name: str,
    agent_name: str,
    agent_description: Optional[str] = None,
    message_path: str = "conversation.messages",
) -> Dict[str, Any]:
    """Create metadata for a node following GenAI semantic conventions.

    Args:
        node_name: The LangGraph node name
        agent_name: Human-readable agent identifier (gen_ai.agent.name)
        agent_description: Optional agent description (gen_ai.agent.description)
        message_path: Dot-path to messages in state (for tracer extraction)

    Returns:
        Metadata dict for the node
    """
    meta = {
        "otel_trace": True,
        "otel_messages_path": message_path,
        "agent_name": agent_name,
        "langgraph_node": node_name,
        "gen_ai.agent.name": agent_name,
    }
    if agent_description:
        meta["gen_ai.agent.description"] = agent_description
    return meta


# =============================================================================
# Workflow 1: Request Classification and Routing
# =============================================================================


def classify_request(state: ComplexAgentState) -> Dict[str, Any]:
    """Classify the user request into a category for routing."""
    llm = create_llm("classifier")
    messages = state.conversation.messages

    classification_prompt = SystemMessage(content="""
Classify the user request into one of these categories:
- TOOL_QUERY: Needs tools like search, weather, calculator, database
- COMPLEX_TASK: Requires orchestration with multiple workers
- SIMPLE_QA: Simple question that can be answered directly

Respond with ONLY the category name.
""")

    response = llm.invoke([classification_prompt] + messages)
    request_type = response.content.strip().upper()

    # Validate and default
    valid_types = {"TOOL_QUERY", "COMPLEX_TASK", "SIMPLE_QA"}
    if request_type not in valid_types:
        request_type = "SIMPLE_QA"

    return {
        "conversation": ConversationThread(
            messages=messages + [AIMessage(content=f"[Classification: {request_type}]")],
            thread_id=state.conversation.thread_id,
        ),
        "request_type": request_type,
    }


def route_by_type(state: ComplexAgentState) -> str:
    """Route to the appropriate handler based on classification."""
    request_type = state.request_type or "SIMPLE_QA"
    routing = {
        "TOOL_QUERY": "tool_agent",
        "COMPLEX_TASK": "orchestrator",
        "SIMPLE_QA": "direct_responder",
    }
    return routing.get(request_type, "direct_responder")


# =============================================================================
# Workflow 2: Tool-Calling ReAct Agent
# =============================================================================


def init_tool_agent(state: ComplexAgentState) -> Dict[str, Any]:
    """Initialize the ReAct tool-calling agent state."""
    return {
        "metadata": {
            **state.metadata,
            "sub_workflow": "tool_agent",
        }
    }


def create_react_agent_graph() -> StateGraph:
    """Create a ReAct agent graph with tool calling."""

    def agent_node(state: ReActState) -> Dict[str, Any]:
        """The agent decides whether to call tools or respond."""
        llm = create_llm("react_agent", temperature=0.1)
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        system = SystemMessage(content="""
You are a helpful assistant with access to tools. Use them when needed.
Available tools: search_web, calculate, get_weather, query_database, retrieve_knowledge
Always provide a final answer after using tools.
""")

        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def should_continue(state: ReActState) -> str:
        """Check if we should continue calling tools or finish."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if state.get("tool_calls_count", 0) < 5:  # Max 5 tool calls
                return "tools"
        return "end"

    # Build the graph
    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node, metadata=agent_metadata(
        "agent", "ReActAgent", "Tool-calling agent using ReAct pattern", "messages"
    ))
    graph.add_node("tools", ToolNode(ALL_TOOLS), metadata={
        "otel_trace": True,
        "langgraph_node": "tools",
    })

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph


# =============================================================================
# Workflow 3: Orchestrator-Worker Pattern with Parallel Execution
# =============================================================================


def orchestrator_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Break down the complex task into subtasks for workers."""
    llm = create_llm("orchestrator")
    messages = state.conversation.messages

    prompt = SystemMessage(content="""
You are an orchestrator. Break down the user's complex request into 3 specific subtasks.
Format your response as:
TASK 1: [description]
TASK 2: [description]
TASK 3: [description]
""")

    response = llm.invoke([prompt] + messages)

    # Parse tasks (simplified)
    tasks = []
    for line in response.content.split("\n"):
        if line.strip().startswith("TASK"):
            tasks.append(line.split(":", 1)[-1].strip())

    if not tasks:
        tasks = ["Research the topic", "Analyze findings", "Synthesize response"]

    return {
        "conversation": ConversationThread(
            messages=messages + [AIMessage(content=f"Orchestrator created {len(tasks)} tasks")],
            thread_id=state.conversation.thread_id,
        ),
        "metadata": {**state.metadata, "tasks": tasks},
    }


async def worker_node(state: ComplexAgentState, worker_id: str, task: str) -> WorkerResult:
    """Execute a single worker task."""
    llm = create_llm(f"worker_{worker_id}")

    prompt = SystemMessage(content=f"""
You are Worker {worker_id}. Complete this specific task:
{task}

Provide a concise, actionable result.
""")

    response = llm.invoke([prompt] + state.conversation.messages[-2:])

    return WorkerResult(
        worker_id=worker_id,
        task=task,
        result=response.content,
        confidence=random.uniform(0.7, 1.0),
    )


def parallel_workers(state: ComplexAgentState) -> Dict[str, Any]:
    """Execute workers in parallel (simulated for sync execution)."""
    tasks = state.metadata.get("tasks", ["Default task"])

    results = []
    for idx, task in enumerate(tasks[:3]):  # Max 3 workers
        llm = create_llm(f"worker_{idx}")
        prompt = SystemMessage(content=f"Complete this task: {task}")
        response = llm.invoke([prompt] + state.conversation.messages[-2:])
        results.append(WorkerResult(
            worker_id=f"W{idx}",
            task=task,
            result=response.content,
            confidence=random.uniform(0.7, 1.0),
        ))

    return {
        "worker_results": results,
        "conversation": ConversationThread(
            messages=state.conversation.messages + [
                AIMessage(content=f"Workers completed {len(results)} tasks")
            ],
            thread_id=state.conversation.thread_id,
        ),
    }


def synthesizer_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Synthesize worker results into a coherent response."""
    llm = create_llm("synthesizer")

    worker_summary = "\n".join([
        f"- {r.worker_id} ({r.task}): {r.result[:100]}..."
        for r in state.worker_results
    ])

    prompt = SystemMessage(content=f"""
Synthesize these worker results into a coherent final response:

{worker_summary}

Create a unified, helpful response for the user.
""")

    response = llm.invoke([prompt] + state.conversation.messages[-3:])

    return {
        "final_response": response.content,
        "conversation": ConversationThread(
            messages=state.conversation.messages + [AIMessage(content=response.content)],
            thread_id=state.conversation.thread_id,
        ),
    }


# =============================================================================
# Workflow 4: Evaluator-Optimizer Loop
# =============================================================================


def direct_responder(state: ComplexAgentState) -> Dict[str, Any]:
    """Generate a direct response for simple questions."""
    llm = create_llm("responder")

    prompt = SystemMessage(content="Provide a helpful, accurate response to the user's question.")
    response = llm.invoke([prompt] + state.conversation.messages)

    return {
        "final_response": response.content,
        "conversation": ConversationThread(
            messages=state.conversation.messages + [AIMessage(content=response.content)],
            thread_id=state.conversation.thread_id,
        ),
    }


def evaluator_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Evaluate the quality of the generated response."""
    llm = create_llm("evaluator", temperature=0.1)

    current_response = state.final_response or ""

    prompt = SystemMessage(content=f"""
Evaluate this response for quality:

Response: {current_response}

Score from 0-10 and provide brief feedback. Format:
SCORE: [number]
FEEDBACK: [brief feedback]
PASS: [YES/NO]
""")

    eval_response = llm.invoke([prompt] + state.conversation.messages[-2:])

    # Parse evaluation
    content = eval_response.content
    score = 7.0  # Default
    feedback = "Response is acceptable"
    passed = True

    for line in content.split("\n"):
        if "SCORE:" in line:
            try:
                score = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "FEEDBACK:" in line:
            feedback = line.split(":")[-1].strip()
        elif "PASS:" in line:
            passed = "YES" in line.upper()

    return {
        "evaluation": EvaluationResult(score=score, feedback=feedback, passed=passed),
        "iteration_count": state.iteration_count + 1,
    }


def should_optimize(state: ComplexAgentState) -> str:
    """Decide whether to optimize the response or finish."""
    if state.evaluation is None:
        return "evaluate"
    if state.evaluation.passed or state.iteration_count >= 3:
        return "finish"
    return "optimize"


def optimizer_node(state: ComplexAgentState) -> Dict[str, Any]:
    """Improve the response based on evaluator feedback."""
    llm = create_llm("optimizer")

    feedback = state.evaluation.feedback if state.evaluation else "Improve clarity"
    current = state.final_response or ""

    prompt = SystemMessage(content=f"""
Improve this response based on feedback:

Current response: {current}
Feedback: {feedback}

Provide an improved response.
""")

    response = llm.invoke([prompt] + state.conversation.messages[-2:])

    return {
        "final_response": response.content,
        "conversation": ConversationThread(
            messages=state.conversation.messages + [
                AIMessage(content=f"[Optimization {state.iteration_count}] {response.content}")
            ],
            thread_id=state.conversation.thread_id,
        ),
    }


# =============================================================================
# Main Graph Assembly
# =============================================================================


def build_complex_agent_graph() -> StateGraph:
    """Build the complete complex multi-workflow agent graph."""

    graph = StateGraph(ComplexAgentState)

    # Add classification node
    graph.add_node("classifier", classify_request, metadata=agent_metadata(
        "classifier", "RequestClassifier",
        "Classifies user requests into categories for routing"
    ))

    # Add tool agent subgraph entry
    def tool_agent_wrapper(state: ComplexAgentState) -> Dict[str, Any]:
        """Wrap the ReAct agent for the main graph."""
        react_graph = create_react_agent_graph().compile(name="react-tools")

        result = react_graph.invoke({
            "messages": state.conversation.messages,
            "thread_id": state.conversation.thread_id,
            "tool_calls_count": 0,
        })

        final_message = result["messages"][-1]
        return {
            "final_response": final_message.content,
            "conversation": ConversationThread(
                messages=state.conversation.messages + [final_message],
                thread_id=state.conversation.thread_id,
            ),
        }

    graph.add_node("tool_agent", tool_agent_wrapper, metadata=agent_metadata(
        "tool_agent", "ToolAgent",
        "Executes tool calls using ReAct pattern"
    ))

    # Add orchestrator-worker nodes
    graph.add_node("orchestrator", orchestrator_node, metadata=agent_metadata(
        "orchestrator", "TaskOrchestrator",
        "Breaks complex tasks into parallel subtasks"
    ))
    graph.add_node("workers", parallel_workers, metadata=agent_metadata(
        "workers", "ParallelWorkers",
        "Executes subtasks in parallel"
    ))
    graph.add_node("synthesizer", synthesizer_node, metadata=agent_metadata(
        "synthesizer", "ResultSynthesizer",
        "Combines worker results into coherent response"
    ))

    # Add direct response path
    graph.add_node("direct_responder", direct_responder, metadata=agent_metadata(
        "direct_responder", "DirectResponder",
        "Handles simple questions directly"
    ))

    # Add evaluator-optimizer loop
    graph.add_node("evaluator", evaluator_node, metadata=agent_metadata(
        "evaluator", "ResponseEvaluator",
        "Evaluates response quality"
    ))
    graph.add_node("optimizer", optimizer_node, metadata=agent_metadata(
        "optimizer", "ResponseOptimizer",
        "Improves responses based on feedback"
    ))

    # Define edges
    graph.add_edge(START, "classifier")

    # Routing from classifier
    graph.add_conditional_edges(
        "classifier",
        route_by_type,
        {
            "tool_agent": "tool_agent",
            "orchestrator": "orchestrator",
            "direct_responder": "direct_responder",
        }
    )

    # Orchestrator-worker flow
    graph.add_edge("orchestrator", "workers")
    graph.add_edge("workers", "synthesizer")
    graph.add_edge("synthesizer", "evaluator")

    # Tool agent to evaluator
    graph.add_edge("tool_agent", "evaluator")

    # Direct responder to evaluator
    graph.add_edge("direct_responder", "evaluator")

    # Evaluator-optimizer loop
    graph.add_conditional_edges(
        "evaluator",
        should_optimize,
        {
            "optimize": "optimizer",
            "finish": END,
            "evaluate": "evaluator",  # Shouldn't happen but safety
        }
    )
    graph.add_edge("optimizer", "evaluator")

    return graph


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the complex multi-workflow agent sample."""

    # Configure tracing using the static autolog() API
    # This automatically traces all LangChain/LangGraph executions
    print("Configuring AzureAIOpenTelemetryTracer with autolog()...")

    # Set up Azure Application Insights if connection string is available
    app_insights_cs = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if app_insights_cs:
        AzureAIOpenTelemetryTracer.set_app_insights(app_insights_cs)
        print(f"  Azure Monitor configured")

    # Configure tracer options
    AzureAIOpenTelemetryTracer.set_config({
        "provider_name": os.getenv("OTEL_GENAI_PROVIDER", "openai"),
        "trace_all_langgraph_nodes": True,
        "message_paths": ("conversation.messages", "messages"),
        "enable_performance_counters": False,
    })

    # Activate autolog - all subsequent LangChain calls are now traced!
    AzureAIOpenTelemetryTracer.autolog()
    print(f"  autolog() activated: {AzureAIOpenTelemetryTracer.is_active()}")

    # Build the graph
    print("\nBuilding complex multi-workflow agent graph...")
    graph = build_complex_agent_graph().compile(name="complex-multi-workflow-agent")

    # Test scenarios
    test_cases = [
        ("What is 25 * 4 + 100?", "TOOL_QUERY - Calculator"),
        ("Plan a weekend trip to Paris with activities and restaurants", "COMPLEX_TASK - Orchestrator"),
        ("What is the capital of France?", "SIMPLE_QA - Direct"),
    ]

    for idx, (query, expected_type) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {idx}: {expected_type}")
        print(f"Query: {query}")
        print("="*60)

        # Add request-specific tags
        AzureAIOpenTelemetryTracer.add_tags({
            "test.case_id": idx,
            "test.expected_type": expected_type,
        })

        # Create initial state
        thread_id = str(uuid4())
        initial_state = ComplexAgentState(
            conversation=ConversationThread(
                messages=[HumanMessage(content=query)],
                thread_id=thread_id,
            ),
            metadata={"test_case": idx},
        )

        # Run the graph
        try:
            result = graph.invoke(initial_state)

            # Extract results
            final_response = result.final_response if hasattr(result, "final_response") else result.get("final_response")
            request_type = result.request_type if hasattr(result, "request_type") else result.get("request_type")
            evaluation = result.evaluation if hasattr(result, "evaluation") else result.get("evaluation")

            print(f"\nClassified as: {request_type}")
            print(f"Final Response: {final_response[:200] if final_response else 'N/A'}...")
            if evaluation:
                print(f"Evaluation: Score={evaluation.score}, Passed={evaluation.passed}")
        except Exception as e:
            print(f"Error: {e}")

    # Flush and shutdown
    print("\n" + "="*60)
    print("Flushing traces and shutting down...")
    AzureAIOpenTelemetryTracer.force_flush()
    AzureAIOpenTelemetryTracer.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
