"""Reusable evaluator-optimizer graph builders for LangGraph.

Provides helper functions to construct evaluator-optimizer subgraphs
that can be embedded as nodes in larger LangGraph workflows. The
pattern follows the LangGraph evaluator-optimizer documentation:
a generator creates output, an evaluator grades it, and a conditional
edge routes back to the generator or exits.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from langchain_azure_ai._api.base import experimental

LOGGER = logging.getLogger(__name__)


@experimental(message="Foundry evaluation integration is in preview and may change.")
def create_eval_optimize_subgraph(
    *,
    evaluate_fn: Callable[[Any], dict[str, Any]],
    refine_fn: Callable[[Any], dict[str, Any]],
    should_refine_fn: Callable[[Any], str],
    state_schema: type,
    max_iterations: int = 3,
    accepted_route: str = "accepted",
    refine_route: str = "refine",
) -> Any:
    """Build a compiled evaluator-optimizer subgraph.

    Creates a LangGraph ``StateGraph`` that implements the
    evaluate→refine loop pattern. The returned compiled graph can
    be used as a node in a parent graph.

    Args:
        evaluate_fn: Node function that evaluates the current draft
            and returns state updates including evaluation results.
        refine_fn: Node function that refines the draft based on
            evaluation feedback.
        should_refine_fn: Routing function that returns
            *accepted_route* or *refine_route* based on state.
        state_schema: The TypedDict class for the subgraph state.
        max_iterations: Safety limit on refinement loops.
        accepted_route: Route name when evaluation passes.
        refine_route: Route name when refinement is needed.

    Returns:
        A compiled LangGraph ``StateGraph`` ready to be invoked
        or added as a node in a parent graph.
    """
    from langgraph.graph import END, START, StateGraph

    builder: Any = StateGraph(state_schema)
    iteration_counter = {"count": 0}

    def should_refine_with_guard(state: Any) -> str:
        route = should_refine_fn(state)
        if route == accepted_route:
            iteration_counter["count"] = 0
            return accepted_route

        iteration_counter["count"] += 1
        if iteration_counter["count"] >= max_iterations:
            iteration_counter["count"] = 0
            return accepted_route

        return route

    builder.add_node("evaluate", cast(Any, evaluate_fn))
    builder.add_node("refine", cast(Any, refine_fn))

    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        should_refine_with_guard,
        {
            accepted_route: END,
            refine_route: "refine",
        },
    )
    builder.add_edge("refine", "evaluate")

    return builder.compile()


@experimental(message="Foundry evaluation integration is in preview and may change.")
def create_analyst_subgraph(
    *,
    research_fn: Callable[[Any], dict[str, Any]],
    write_fn: Callable[[Any], dict[str, Any]],
    eval_optimize_graph: Any,
    build_completed_fn: Callable[[Any], dict[str, Any]],
    state_schema: type,
    max_iterations: int = 3,
) -> Any:
    """Build a compiled analyst subgraph with embedded eval-optimize loop.

    Creates a LangGraph ``StateGraph`` for a specialist analyst:
    research → write → eval-optimize (subgraph) → build_completed.

    The eval-optimize loop is invoked as a subgraph with a different
    state schema, demonstrating the parent→child→grandchild pattern
    with state transformation.

    Args:
        research_fn: Node that gathers research data.
        write_fn: Node that writes the section draft.
        eval_optimize_graph: Compiled eval-optimize subgraph.
        build_completed_fn: Node that packages final output.
        state_schema: The analyst TypedDict state.
        max_iterations: Safety limit passed to the eval-optimize subgraph.

    Returns:
        A compiled LangGraph ``StateGraph``.
    """
    from langgraph.graph import END, START, StateGraph

    def call_eval_optimize(state: dict[str, Any]) -> dict[str, Any]:
        """Transform analyst state → eval state, invoke, transform back."""
        eval_input = {
            "section_area": (
                state.get("section", {}).area
                if hasattr(state.get("section"), "area")
                else "unknown"
            ),
            "section_title": (
                state.get("section", {}).title
                if hasattr(state.get("section"), "title")
                else "untitled"
            ),
            "draft_content": state.get("draft_content", ""),
            "evaluation_feedback": "",
            "evaluation_result": None,
            "accepted": False,
            "iteration": 0,
            "max_iterations": max_iterations,
        }

        eval_output = eval_optimize_graph.invoke(eval_input)

        return {
            "draft_content": eval_output.get(
                "draft_content", state.get("draft_content", "")
            ),
            "evaluation_result": eval_output.get("evaluation_result"),
            "iteration_count": eval_output.get("iteration", 0),
        }

    builder: Any = StateGraph(state_schema)

    builder.add_node("research", cast(Any, research_fn))
    builder.add_node("write", cast(Any, write_fn))
    builder.add_node("eval_optimize", call_eval_optimize)
    builder.add_node("build_completed", cast(Any, build_completed_fn))

    builder.add_edge(START, "research")
    builder.add_edge("research", "write")
    builder.add_edge("write", "eval_optimize")
    builder.add_edge("eval_optimize", "build_completed")
    builder.add_edge("build_completed", END)

    return builder.compile()
