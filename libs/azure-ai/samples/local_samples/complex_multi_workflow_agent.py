"""Complex Multi-Workflow Agent Sample with Deep Nested Hierarchies.

This sample demonstrates complex parent-child span relationships:

SPAN HIERARCHY EXAMPLE:
invoke_agent (SupervisorAgent)
├── chat (Supervisor decides to delegate)
│   └── execute_tool (delegate_to_research_agent)
│       └── invoke_agent (ResearchAgent - nested agent!)
│           ├── chat (Research agent decides to search)
│           │   └── execute_tool (search_web)
│           ├── chat (Research agent decides to query DB)
│           │   └── execute_tool (query_database)
│           └── chat (Research agent synthesizes)
├── chat (Supervisor decides to use analysis agent)
│   └── execute_tool (delegate_to_analysis_agent)
│       └── invoke_agent (AnalysisAgent - another nested agent!)
│           ├── chat (Analysis uses calculator)
│           │   └── execute_tool (calculate)
│           └── chat (Analysis produces result)
└── chat (Supervisor produces final response)

Key Features:
1. **Tool-as-Agent Pattern** - Tools that spawn entire sub-agent workflows
2. **Multi-level Nesting** - Supervisor -> Sub-Agent -> Chat -> Tool chains
3. **Parallel Agent Execution** - Multiple sub-agents working concurrently
4. **Recursive Tool Calls** - Tools that can call other tools/agents
5. **Complex State Flow** - State passed through multiple agent boundaries

Run with Python from repo root venv:
  .venv-py311/bin/python libs/azure-ai/samples/local_samples/complex_multi_workflow_agent.py

Environment variables:
  AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_COMPLETIONS_URL (or components)
  OPENAI_API_KEY for public OpenAI
  APPLICATIONINSIGHTS_CONNECTION_STRING for Azure Monitor (optional)
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Annotated, TypedDict
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from operator import add

from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer


# =============================================================================
# LLM Factory
# =============================================================================


class OfflineLLM:
    """Offline fallback LLM for testing without API keys."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._bound_tools: List[Any] = []

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "OfflineLLM":
        new_llm = OfflineLLM(self.agent_name)
        new_llm._bound_tools = tools
        return new_llm

    def invoke(self, messages: List[Any], **kwargs: Any) -> AIMessage:
        # Check if last message is a tool response - if so, synthesize an answer
        last_msg = messages[-1] if messages else None
        is_tool_response = hasattr(last_msg, "type") and last_msg.type == "tool"

        # Count existing tool calls to avoid infinite loops
        tool_call_count = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)

        # If we've had tool responses or too many tool calls, give a final answer
        if is_tool_response or tool_call_count >= 2:
            return AIMessage(content=f"[{self.agent_name}] Based on the gathered information, here is my analysis and conclusion for the requested task.")

        last_content = last_msg.content if last_msg and hasattr(last_msg, "content") else "Request"
        last_lower = last_content.lower()

        # Simulate tool calls based on content
        if self._bound_tools:
            tool_names = [t.name for t in self._bound_tools]

            # Supervisor delegating to research
            if "delegate_to_research_agent" in tool_names and ("research" in last_lower or "search" in last_lower or "find" in last_lower):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "delegate_to_research_agent",
                        "id": f"call_{uuid4().hex[:8]}",
                        "args": {"query": last_content, "depth": "thorough"},
                    }]
                )

            # Supervisor delegating to analysis
            if "delegate_to_analysis_agent" in tool_names and ("analyze" in last_lower or "calculate" in last_lower or "compare" in last_lower):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "delegate_to_analysis_agent",
                        "id": f"call_{uuid4().hex[:8]}",
                        "args": {"task": last_content, "metrics": ["accuracy", "relevance"]},
                    }]
                )

            # Supervisor delegating to writer
            if "delegate_to_writer_agent" in tool_names and ("write" in last_lower or "draft" in last_lower or "compose" in last_lower):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "delegate_to_writer_agent",
                        "id": f"call_{uuid4().hex[:8]}",
                        "args": {"topic": last_content, "style": "professional"},
                    }]
                )

            # Research agent using search
            if "search_web" in tool_names and "delegate_to_research_agent" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "search_web",
                        "id": f"call_{uuid4().hex[:8]}",
                        "args": {"query": "relevant information about " + last_content[:50]},
                    }]
                )

            # Analysis agent using calculator
            if "calculate" in tool_names and "delegate_to_analysis_agent" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "calculate",
                        "id": f"call_{uuid4().hex[:8]}",
                        "args": {"expression": "100 * 1.5 + 50"},
                    }]
                )

            # Writer agent using knowledge retrieval
            if "retrieve_knowledge" in tool_names and "delegate_to_writer_agent" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "retrieve_knowledge",
                        "id": f"call_{uuid4().hex[:8]}",
                        "args": {"topic": "writing guidelines", "max_results": 3},
                    }]
                )

        return AIMessage(content=f"[{self.agent_name}] Completed task: {last_content[:100]}...")


def create_llm(agent_name: str, temperature: float = 0.3) -> ChatOpenAI:
    """Create an LLM instance, preferring Azure OpenAI if configured."""

    def parse_azure_url(url: str) -> tuple[str, str, Optional[str]]:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        parts = [p for p in parsed.path.split("/") if p]
        deployment = None
        for i, p in enumerate(parts):
            if p == "deployments" and i + 1 < len(parts):
                deployment = parts[i + 1]
                break
        api_ver = parse_qs(parsed.query).get("api-version", [None])[0]
        if not base or not deployment:
            raise ValueError(f"Invalid Azure OpenAI URL: {url}")
        return base, deployment, api_ver

    azure_url = os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_URL") or os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_ENDPOINT")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_BASE_URL")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")

    if azure_url:
        azure_endpoint, deployment, api_ver = parse_azure_url(azure_url)
        api_version = api_ver or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
    elif azure_endpoint and "/openai/deployments/" in azure_endpoint:
        azure_endpoint, deployment, api_ver = parse_azure_url(azure_endpoint)
        api_version = api_ver or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
    else:
        azure_endpoint = azure_endpoint.rstrip("/") if azure_endpoint else azure_endpoint
        deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or
            os.getenv("AZURE_OPENAI_DEPLOYMENT") or
            os.getenv("OPENAI_MODEL")
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"

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

    saved = {k: os.environ.pop(k) for k in ["OPENAI_API_VERSION", "OPENAI_BASE_URL"] if k in os.environ}
    try:
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1"), temperature=temperature, max_tokens=500)
    finally:
        os.environ.update(saved)


# =============================================================================
# Basic Tools - Used by Sub-Agents
# =============================================================================


@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results
    """
    results = {
        "weather": "Current weather: Sunny, 72°F with clear skies.",
        "news": "Breaking: AI technology advancing rapidly in 2024.",
        "travel": "Popular destinations: Paris, Tokyo, Barcelona.",
        "food": "Top restaurants feature farm-to-table cuisine.",
        "tech": "Latest trends: AI agents, multimodal models, edge computing.",
    }
    for key, value in results.items():
        if key in query.lower():
            return f"[SearchResult] {value}"
    return f"[SearchResult] Found 15 results for '{query}'. Top result: Comprehensive coverage available."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression (e.g., '2 + 2 * 3')

    Returns:
        Calculation result
    """
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "[CalcError] Invalid characters"
        result = eval(expression)
        return f"[CalcResult] {expression} = {result}"
    except Exception as e:
        return f"[CalcError] {str(e)}"


@tool
def query_database(table: str, filters: str = "") -> str:
    """Query a database table.

    Args:
        table: Table name
        filters: Optional filters (e.g., 'status=active')

    Returns:
        Query results as JSON
    """
    data = {
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "orders": [{"id": 101, "total": 150.00}, {"id": 102, "total": 89.99}],
        "products": [{"id": "A1", "name": "Widget", "price": 29.99}],
    }
    rows = data.get(table, [])
    return f"[DBResult] Table '{table}': {len(rows)} rows. Data: {json.dumps(rows[:2])}"


@tool
def retrieve_knowledge(topic: str, max_results: int = 3) -> str:
    """Retrieve knowledge base articles.

    Args:
        topic: Topic to search
        max_results: Max results to return

    Returns:
        Knowledge base articles
    """
    kb = {
        "writing": ["Use active voice", "Keep sentences concise", "Structure with headers"],
        "technical": ["API rate limit: 1000/min", "Use OAuth 2.0", "SDK available"],
        "policy": ["30-day returns", "5-7 day refunds", "Contact support for warranty"],
    }
    for key, articles in kb.items():
        if key in topic.lower():
            return f"[KBResult] Found {len(articles)} articles on '{topic}': {articles[:max_results]}"
    return f"[KBResult] No specific articles for '{topic}'. General guidelines apply."


@tool
def get_weather(location: str) -> str:
    """Get weather for a location.

    Args:
        location: City name

    Returns:
        Weather information
    """
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    temp = random.randint(60, 85)
    return f"[Weather] {location}: {random.choice(conditions)}, {temp}°F, Humidity: {random.randint(40, 80)}%"


# =============================================================================
# Sub-Agent Graphs - These are invoked by delegation tools
# =============================================================================


class SubAgentState(TypedDict):
    """State for sub-agent workflows."""
    messages: Annotated[List[BaseMessage], add]
    tool_calls_count: int


def create_research_agent() -> StateGraph:
    """Create the Research Agent sub-graph.

    This agent specializes in gathering information using search and database tools.
    It produces a nested span hierarchy:

    invoke_agent (ResearchAgent)
    ├── chat (decides to search)
    │   └── execute_tool (search_web)
    ├── chat (decides to query DB)
    │   └── execute_tool (query_database)
    └── chat (synthesizes findings)
    """
    research_tools = [search_web, query_database, get_weather]

    def research_agent_node(state: SubAgentState) -> Dict[str, Any]:
        """Research agent reasoning node."""
        llm = create_llm("research_agent", temperature=0.2)
        llm_with_tools = llm.bind_tools(research_tools)

        system = SystemMessage(content="""You are a Research Agent specialized in gathering information.
You have access to: search_web, query_database, get_weather.
Use multiple tools to gather comprehensive information.
After gathering data, synthesize your findings into a clear summary.""")

        state_messages = state.get("messages", [])
        messages = [system] + state_messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}  # Will be added via reducer

    def should_continue(state: SubAgentState) -> str:
        state_messages = state.get("messages", [])
        last = state_messages[-1] if state_messages else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            if state.get("tool_calls_count", 0) < 4:
                return "tools"
        return "end"

    def count_tool_calls(state: SubAgentState) -> Dict[str, Any]:
        return {"tool_calls_count": state.get("tool_calls_count", 0) + 1}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", research_agent_node, metadata={
        "otel_trace": True,
        "agent_name": "ResearchAgent",
        "otel_messages_path": "messages",
        "gen_ai.agent.name": "ResearchAgent",
        "gen_ai.agent.description": "Gathers information using search and database tools",
    })
    graph.add_node("tools", ToolNode(research_tools), metadata={
        "otel_trace": True,
        "langgraph_node": "research_tools",
    })
    graph.add_node("count", count_tool_calls)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "count")
    graph.add_edge("count", "agent")

    return graph


def create_analysis_agent() -> StateGraph:
    """Create the Analysis Agent sub-graph.

    This agent specializes in calculations and data analysis.
    Produces spans:

    invoke_agent (AnalysisAgent)
    ├── chat (decides to calculate)
    │   └── execute_tool (calculate)
    ├── chat (another calculation)
    │   └── execute_tool (calculate)
    └── chat (produces analysis)
    """
    analysis_tools = [calculate, query_database]

    def analysis_agent_node(state: SubAgentState) -> Dict[str, Any]:
        """Analysis agent reasoning node."""
        llm = create_llm("analysis_agent", temperature=0.1)
        llm_with_tools = llm.bind_tools(analysis_tools)

        system = SystemMessage(content="""You are an Analysis Agent specialized in calculations and data analysis.
You have access to: calculate, query_database.
Perform calculations and analyze data to produce insights.
Provide quantitative analysis where possible.""")

        state_messages = state.get("messages", [])
        messages = [system] + state_messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}  # Will be added via reducer

    def should_continue_analysis(state: SubAgentState) -> str:
        state_messages = state.get("messages", [])
        last = state_messages[-1] if state_messages else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            if state.get("tool_calls_count", 0) < 3:
                return "tools"
        return "end"

    def count_tool_calls_analysis(state: SubAgentState) -> Dict[str, Any]:
        return {"tool_calls_count": state.get("tool_calls_count", 0) + 1}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", analysis_agent_node, metadata={
        "otel_trace": True,
        "agent_name": "AnalysisAgent",
        "otel_messages_path": "messages",
        "gen_ai.agent.name": "AnalysisAgent",
        "gen_ai.agent.description": "Performs calculations and data analysis",
    })
    graph.add_node("tools", ToolNode(analysis_tools), metadata={
        "otel_trace": True,
        "langgraph_node": "analysis_tools",
    })
    graph.add_node("count", count_tool_calls_analysis)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue_analysis, {"tools": "tools", "end": END})
    graph.add_edge("tools", "count")
    graph.add_edge("count", "agent")

    return graph


def create_writer_agent() -> StateGraph:
    """Create the Writer Agent sub-graph.

    This agent specializes in content creation using knowledge retrieval.
    Produces spans:

    invoke_agent (WriterAgent)
    ├── chat (retrieves knowledge)
    │   └── execute_tool (retrieve_knowledge)
    └── chat (produces content)
    """
    writer_tools = [retrieve_knowledge, search_web]

    def writer_agent_node(state: SubAgentState) -> Dict[str, Any]:
        """Writer agent reasoning node."""
        llm = create_llm("writer_agent", temperature=0.5)
        llm_with_tools = llm.bind_tools(writer_tools)

        system = SystemMessage(content="""You are a Writer Agent specialized in content creation.
You have access to: retrieve_knowledge, search_web.
First gather relevant information, then create well-structured content.
Focus on clarity and engagement.""")

        state_messages = state.get("messages", [])
        messages = [system] + state_messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}  # Will be added via reducer

    def should_continue_writer(state: SubAgentState) -> str:
        state_messages = state.get("messages", [])
        last = state_messages[-1] if state_messages else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            if state.get("tool_calls_count", 0) < 2:
                return "tools"
        return "end"

    def count_tool_calls_writer(state: SubAgentState) -> Dict[str, Any]:
        return {"tool_calls_count": state.get("tool_calls_count", 0) + 1}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", writer_agent_node, metadata={
        "otel_trace": True,
        "agent_name": "WriterAgent",
        "otel_messages_path": "messages",
        "gen_ai.agent.name": "WriterAgent",
        "gen_ai.agent.description": "Creates content using knowledge retrieval",
    })
    graph.add_node("tools", ToolNode(writer_tools), metadata={
        "otel_trace": True,
        "langgraph_node": "writer_tools",
    })
    graph.add_node("count", count_tool_calls_writer)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue_writer, {"tools": "tools", "end": END})
    graph.add_edge("tools", "count")
    graph.add_edge("count", "agent")

    return graph


# =============================================================================
# Delegation Tools - These spawn sub-agent workflows
# =============================================================================


# Compiled sub-agent graphs (created once, reused)
_RESEARCH_GRAPH = None
_ANALYSIS_GRAPH = None
_WRITER_GRAPH = None


def _get_research_graph():
    global _RESEARCH_GRAPH
    if _RESEARCH_GRAPH is None:
        _RESEARCH_GRAPH = create_research_agent().compile(name="research-agent")
    return _RESEARCH_GRAPH


def _get_analysis_graph():
    global _ANALYSIS_GRAPH
    if _ANALYSIS_GRAPH is None:
        _ANALYSIS_GRAPH = create_analysis_agent().compile(name="analysis-agent")
    return _ANALYSIS_GRAPH


def _get_writer_graph():
    global _WRITER_GRAPH
    if _WRITER_GRAPH is None:
        _WRITER_GRAPH = create_writer_agent().compile(name="writer-agent")
    return _WRITER_GRAPH


@tool
def delegate_to_research_agent(query: str, depth: str = "standard") -> str:
    """Delegate a research task to the specialized Research Agent.

    This tool spawns an entire sub-agent workflow that will:
    1. Search the web for relevant information
    2. Query databases for structured data
    3. Synthesize findings into a comprehensive report

    Args:
        query: The research query or topic to investigate
        depth: Research depth - 'quick', 'standard', or 'thorough'

    Returns:
        Research findings from the sub-agent
    """
    print(f"  [DelegationTool] Spawning ResearchAgent for: {query[:50]}...")

    research_graph = _get_research_graph()
    result = research_graph.invoke({
        "messages": [HumanMessage(content=f"Research this topic ({depth} depth): {query}")],
        "tool_calls_count": 0,
    })

    # Extract final response
    final_msg = result["messages"][-1] if result["messages"] else None
    if final_msg:
        return f"[ResearchAgent Result] {final_msg.content}"
    return "[ResearchAgent] Research completed but no summary produced."


@tool
def delegate_to_analysis_agent(task: str, metrics: List[str] = None) -> str:
    """Delegate an analysis task to the specialized Analysis Agent.

    This tool spawns a sub-agent workflow that will:
    1. Perform calculations as needed
    2. Query databases for data
    3. Analyze and produce quantitative insights

    Args:
        task: The analysis task description
        metrics: List of metrics to calculate (e.g., ['accuracy', 'cost'])

    Returns:
        Analysis results from the sub-agent
    """
    metrics = metrics or ["general"]
    print(f"  [DelegationTool] Spawning AnalysisAgent for: {task[:50]}...")

    analysis_graph = _get_analysis_graph()
    result = analysis_graph.invoke({
        "messages": [HumanMessage(content=f"Analyze this (metrics: {metrics}): {task}")],
        "tool_calls_count": 0,
    })

    final_msg = result["messages"][-1] if result["messages"] else None
    if final_msg:
        return f"[AnalysisAgent Result] {final_msg.content}"
    return "[AnalysisAgent] Analysis completed but no summary produced."


@tool
def delegate_to_writer_agent(topic: str, style: str = "professional") -> str:
    """Delegate a writing task to the specialized Writer Agent.

    This tool spawns a sub-agent workflow that will:
    1. Retrieve relevant knowledge
    2. Search for additional context
    3. Produce well-structured content

    Args:
        topic: The topic or subject to write about
        style: Writing style - 'professional', 'casual', 'technical'

    Returns:
        Written content from the sub-agent
    """
    print(f"  [DelegationTool] Spawning WriterAgent for: {topic[:50]}...")

    writer_graph = _get_writer_graph()
    result = writer_graph.invoke({
        "messages": [HumanMessage(content=f"Write about this ({style} style): {topic}")],
        "tool_calls_count": 0,
    })

    final_msg = result["messages"][-1] if result["messages"] else None
    if final_msg:
        return f"[WriterAgent Result] {final_msg.content}"
    return "[WriterAgent] Writing completed but no content produced."


# All delegation tools for the supervisor
DELEGATION_TOOLS = [
    delegate_to_research_agent,
    delegate_to_analysis_agent,
    delegate_to_writer_agent,
]

# Direct tools the supervisor can also use
DIRECT_TOOLS = [search_web, calculate, get_weather]

# All tools available to supervisor
SUPERVISOR_TOOLS = DELEGATION_TOOLS + DIRECT_TOOLS


# =============================================================================
# Supervisor Agent - Orchestrates Sub-Agents
# =============================================================================


class SupervisorState(TypedDict):
    """State for the Supervisor Agent workflow."""
    messages: Annotated[List[BaseMessage], add]
    tool_calls_count: int
    delegations_made: List[str]


def create_supervisor_agent() -> StateGraph:
    """Create the Supervisor Agent that orchestrates sub-agents.

    This is the top-level agent that:
    1. Analyzes user requests
    2. Delegates to specialized sub-agents via tools
    3. Coordinates multiple sub-agent results
    4. Produces final synthesized responses

    Full span hierarchy example:

    invoke_agent (SupervisorAgent)
    ├── chat (analyzes request, decides to delegate)
    │   └── execute_tool (delegate_to_research_agent)
    │       └── invoke_agent (ResearchAgent)
    │           ├── chat → execute_tool (search_web)
    │           ├── chat → execute_tool (query_database)
    │           └── chat (synthesize)
    ├── chat (reviews research, needs analysis)
    │   └── execute_tool (delegate_to_analysis_agent)
    │       └── invoke_agent (AnalysisAgent)
    │           ├── chat → execute_tool (calculate)
    │           └── chat (produce analysis)
    ├── chat (needs content written)
    │   └── execute_tool (delegate_to_writer_agent)
    │       └── invoke_agent (WriterAgent)
    │           ├── chat → execute_tool (retrieve_knowledge)
    │           └── chat (produce content)
    └── chat (final synthesis response)
    """

    def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
        """Supervisor agent reasoning and delegation node."""
        llm = create_llm("supervisor_agent", temperature=0.3)
        llm_with_tools = llm.bind_tools(SUPERVISOR_TOOLS)

        system = SystemMessage(content="""You are a Supervisor Agent that orchestrates specialized sub-agents.

Available delegation tools (spawn sub-agent workflows):
- delegate_to_research_agent: For information gathering, searches, database queries
- delegate_to_analysis_agent: For calculations, data analysis, quantitative tasks
- delegate_to_writer_agent: For content creation, documentation, summaries

Available direct tools:
- search_web: Quick web search
- calculate: Simple math
- get_weather: Weather lookup

Strategy:
1. Analyze the user request
2. Delegate complex sub-tasks to appropriate specialist agents
3. Use multiple agents if needed for comprehensive responses
4. Synthesize all results into a final coherent response

Be strategic - use delegation for complex tasks, direct tools for simple ones.""")

        state_messages = state.get("messages", [])
        messages = [system] + state_messages
        response = llm_with_tools.invoke(messages)

        # Track delegations
        delegations = list(state.get("delegations_made", []))
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"].startswith("delegate_to_"):
                    delegations.append(tc["name"])

        return {
            "messages": [response],  # Will be added via reducer
            "delegations_made": delegations,
        }

    def should_continue(state: SupervisorState) -> str:
        """Determine if supervisor should continue or finish."""
        state_messages = state.get("messages", [])
        last = state_messages[-1] if state_messages else None

        # If last message has tool calls, execute them
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            # Allow more iterations for complex workflows
            if state.get("tool_calls_count", 0) < 8:
                return "tools"

        return "end"

    def count_tool_calls(state: SupervisorState) -> Dict[str, Any]:
        """Track tool call iterations."""
        return {"tool_calls_count": state.get("tool_calls_count", 0) + 1}

    # Build the graph
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node, metadata={
        "otel_trace": True,
        "agent_name": "SupervisorAgent",
        "otel_messages_path": "messages",
        "gen_ai.agent.name": "SupervisorAgent",
        "gen_ai.agent.description": "Orchestrates specialized sub-agents for complex tasks",
    })

    graph.add_node("tools", ToolNode(SUPERVISOR_TOOLS), metadata={
        "otel_trace": True,
        "langgraph_node": "supervisor_tools",
    })

    graph.add_node("count", count_tool_calls)

    # Edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "count")
    graph.add_edge("count", "supervisor")

    return graph


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run the complex multi-workflow agent with deep nesting."""

    print("=" * 70)
    print("Complex Multi-Workflow Agent - Deep Nested Hierarchy Demo")
    print("=" * 70)

    # Configure tracing
    print("\nConfiguring AzureAIOpenTelemetryTracer...")

    app_insights_cs = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if app_insights_cs:
        AzureAIOpenTelemetryTracer.set_app_insights(app_insights_cs)
        print("  ✓ Azure Monitor configured")

    AzureAIOpenTelemetryTracer.set_config({
        "provider_name": os.getenv("OTEL_GENAI_PROVIDER", "azure.openai"),
        "trace_all_langgraph_nodes": True,
        "message_paths": ("messages",),
        "enable_performance_counters": False,
    })

    AzureAIOpenTelemetryTracer.autolog()
    print(f"  ✓ autolog() activated: {AzureAIOpenTelemetryTracer.is_active()}")

    # Build supervisor graph
    print("\nBuilding Supervisor Agent graph with sub-agent delegation...")
    supervisor_graph = create_supervisor_agent().compile(name="supervisor-agent")

    # Test scenarios that exercise different agent hierarchies
    test_cases = [
        {
            "query": "Research the latest AI trends and analyze their market impact",
            "expected": "ResearchAgent + AnalysisAgent delegation",
            "description": "Multi-agent: Research → Analysis pipeline",
        },
        {
            "query": "Calculate the compound interest on $10000 at 5% for 10 years and write a summary report",
            "expected": "AnalysisAgent + WriterAgent delegation",
            "description": "Multi-agent: Analysis → Writer pipeline",
        },
        {
            "query": "Find information about Paris, analyze travel costs, and draft an itinerary",
            "expected": "All three sub-agents",
            "description": "Full delegation: Research + Analysis + Writer",
        },
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test Case {idx}: {test['description']}")
        print(f"Expected: {test['expected']}")
        print(f"Query: {test['query']}")
        print("=" * 70)

        # Add request tags
        AzureAIOpenTelemetryTracer.add_tags({
            "test.case_id": str(idx),
            "test.description": test["description"],
        })

        # Create initial state
        initial_state = SupervisorState({
            "messages": [HumanMessage(content=test["query"])],
            "tool_calls_count": 0,
            "delegations_made": [],
        })

        try:
            # Run the supervisor
            result = supervisor_graph.invoke(initial_state)

            # Extract results
            final_msg = result["messages"][-1] if result["messages"] else None
            delegations = result.get("delegations_made", [])
            tool_calls = result.get("tool_calls_count", 0)

            print(f"\n--- Results ---")
            print(f"Delegations made: {delegations}")
            print(f"Total tool calls: {tool_calls}")
            print(f"Final response: {final_msg.content[:300] if final_msg else 'N/A'}...")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary of span hierarchy
    print("\n" + "=" * 70)
    print("EXPECTED SPAN HIERARCHY (per test case):")
    print("=" * 70)
    print("""
Test 1 (Research + Analysis):
  invoke_agent (SupervisorAgent)
  ├── chat (decides to delegate research)
  │   └── execute_tool (delegate_to_research_agent)
  │       └── invoke_agent (ResearchAgent)
  │           ├── chat → execute_tool (search_web)
  │           └── chat (synthesize)
  ├── chat (decides to delegate analysis)
  │   └── execute_tool (delegate_to_analysis_agent)
  │       └── invoke_agent (AnalysisAgent)
  │           ├── chat → execute_tool (calculate)
  │           └── chat (produce analysis)
  └── chat (final synthesis)

Test 2 (Analysis + Writer):
  invoke_agent (SupervisorAgent)
  ├── chat → execute_tool (delegate_to_analysis_agent)
  │   └── invoke_agent (AnalysisAgent) → chat → execute_tool (calculate)
  ├── chat → execute_tool (delegate_to_writer_agent)
  │   └── invoke_agent (WriterAgent) → chat → execute_tool (retrieve_knowledge)
  └── chat (final synthesis)

Test 3 (All Three):
  invoke_agent (SupervisorAgent)
  ├── chat → execute_tool (delegate_to_research_agent)
  │   └── invoke_agent (ResearchAgent) ...
  ├── chat → execute_tool (delegate_to_analysis_agent)
  │   └── invoke_agent (AnalysisAgent) ...
  ├── chat → execute_tool (delegate_to_writer_agent)
  │   └── invoke_agent (WriterAgent) ...
  └── chat (final synthesis)
""")

    # Flush and shutdown
    print("\nFlushing traces and shutting down...")
    AzureAIOpenTelemetryTracer.force_flush()
    AzureAIOpenTelemetryTracer.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
