"""Main LangGraph graph for the Due Diligence Report Generator.

Implements a 3-level hierarchy:
- Parent: Orchestrator (planner → Send workers → synthesizer → quality gate)
- Child: Analyst subgraphs (research → write → eval-optimize)
- Grandchild: Eval-optimize subgraphs (evaluate → conditional refine loop)

Demonstrates:
- Send API for dynamic parallel worker spawning
- Subgraphs with different state schemas at each level
- Evaluator-optimizer pattern with conditional routing
- State transformation between graph levels
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import (
    build_completed_section,
    evaluate_section,
    final_evaluator,
    planner,
    refine_section,
    research,
    should_refine,
    synthesizer,
    write_section,
)
from eval_config import get_eval_config
from state import (
    AnalystState,
    CompletedSection,
    EvalOptimizeState,
    OrchestratorState,
    WorkerInput,
)


# ============================================================
# Level 3 (Grandchild): Eval-Optimize Subgraph
# ============================================================


def build_eval_optimize_graph() -> Any:
    """Build the evaluator-optimizer subgraph.

    This is the innermost (grandchild) graph that implements the
    evaluate→refine loop. It has its own state schema
    (EvalOptimizeState) separate from the analyst state.

    Flow: evaluate → conditional(accept or refine) → evaluate → ...
    """
    builder = StateGraph(EvalOptimizeState)

    builder.add_node("evaluate", evaluate_section)
    builder.add_node("refine", refine_section)

    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        should_refine,
        {
            "accepted": END,
            "refine": "refine",
        },
    )
    builder.add_edge("refine", "evaluate")

    return builder.compile()


# ============================================================
# Level 2 (Child): Analyst Subgraph
# ============================================================


def build_analyst_graph(eval_optimize_graph: Any) -> Any:
    """Build a specialist analyst subgraph.

    This is the middle (child) level graph. It has its own state schema
    (AnalystState) and invokes the eval-optimize grandchild subgraph
    via a wrapper node that handles state transformation.

    Flow: research → write → eval_optimize (subgraph) → build_completed
    """

    def call_eval_optimize(state: AnalystState) -> dict[str, Any]:
        """Invoke the eval-optimize subgraph with a transformed state."""
        section = state["section"]

        max_iterations = 3
        try:
            config = get_eval_config(section.analyst_type)
            if config.evaluators:
                max_iterations = config.evaluators[0].max_iterations
        except Exception:
            pass

        eval_input: EvalOptimizeState = {
            "section_area": section.area,
            "section_title": section.title,
            "draft_content": state["draft_content"],
            "evaluation_feedback": "",
            "evaluation_result": None,
            "accepted": False,
            "iteration": 0,
            "max_iterations": max_iterations,
        }

        eval_output = eval_optimize_graph.invoke(eval_input)

        return {
            "draft_content": eval_output["draft_content"],
            "evaluation_result": eval_output.get("evaluation_result"),
            "iteration_count": eval_output.get("iteration", 0),
        }

    builder = StateGraph(AnalystState)

    builder.add_node("research", research)
    builder.add_node("write", write_section)
    builder.add_node("eval_optimize", call_eval_optimize)
    builder.add_node("build_completed", build_completed_section)

    builder.add_edge(START, "research")
    builder.add_edge("research", "write")
    builder.add_edge("write", "eval_optimize")
    builder.add_edge("eval_optimize", "build_completed")
    builder.add_edge("build_completed", END)

    return builder.compile()


# ============================================================
# Level 1 (Parent): Orchestrator Graph
# ============================================================


def build_orchestrator_graph(*, parallel: bool = True) -> Any:
    """Build the top-level orchestrator graph.

    Args:
        parallel: If True, use Send API for parallel analyst workers
            (requires high-throughput LLM tier). If False, process
            analysts sequentially (works with S0/rate-limited tiers).
    """
    eval_optimize_graph = build_eval_optimize_graph()
    analyst_graph = build_analyst_graph(eval_optimize_graph)

    def analyst_worker(state: WorkerInput) -> dict[str, Any]:
        """Invoke the analyst subgraph for a single section."""
        section = state["section"]

        analyst_input: AnalystState = {
            "section": section,
            "research_data": [],
            "draft_content": "",
            "evaluation_result": None,
            "completed_section": None,
            "iteration_count": 0,
        }

        analyst_output = analyst_graph.invoke(analyst_input)

        completed = analyst_output.get("completed_section")
        if completed is None:
            completed = CompletedSection(
                area=section.area,
                title=section.title,
                content=analyst_output.get("draft_content", ""),
                research_data=analyst_output.get("research_data", []),
                iterations=analyst_output.get("iteration_count", 0),
            )

        return {"completed_sections": [completed]}

    if parallel:
        # --- Parallel mode: Send API dispatches workers concurrently ---
        def assign_analysts(state: OrchestratorState) -> list[Send]:
            """Dispatch one analyst worker per planned section."""
            return [
                Send(
                    "analyst_worker",
                    WorkerInput(
                        section=section,
                        company_name=state["company_name"],
                        industry=state["industry"],
                    ),
                )
                for section in state["analysis_plan"]
            ]

        builder = StateGraph(OrchestratorState)
        builder.add_node("planner", planner)
        builder.add_node("analyst_worker", analyst_worker)
        builder.add_node("synthesizer", synthesizer)
        builder.add_node("final_evaluator", final_evaluator)

        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", assign_analysts, ["analyst_worker"])
        builder.add_edge("analyst_worker", "synthesizer")
        builder.add_edge("synthesizer", "final_evaluator")
        builder.add_edge("final_evaluator", END)

    else:
        # --- Sequential mode: process analysts one at a time ---
        def run_all_analysts(state: OrchestratorState) -> dict[str, Any]:
            """Run each analyst sequentially to stay under rate limits."""
            completed: list[CompletedSection] = []
            for section in state["analysis_plan"]:
                worker_input = WorkerInput(
                    section=section,
                    company_name=state["company_name"],
                    industry=state["industry"],
                )
                result = analyst_worker(worker_input)
                completed.extend(result["completed_sections"])
            return {"completed_sections": completed}

        builder = StateGraph(OrchestratorState)
        builder.add_node("planner", planner)
        builder.add_node("run_all_analysts", run_all_analysts)
        builder.add_node("synthesizer", synthesizer)
        builder.add_node("final_evaluator", final_evaluator)

        builder.add_edge(START, "planner")
        builder.add_edge("planner", "run_all_analysts")
        builder.add_edge("run_all_analysts", "synthesizer")
        builder.add_edge("synthesizer", "final_evaluator")
        builder.add_edge("final_evaluator", END)

    return builder.compile()


# ============================================================
# Convenience: build the full graph
# ============================================================


def create_due_diligence_graph(*, parallel: bool = True) -> Any:
    """Create the complete due diligence report generator graph.

    Args:
        parallel: Use parallel Send API (True) or sequential mode (False).
    """
    return build_orchestrator_graph(parallel=parallel)
