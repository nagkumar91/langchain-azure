"""Agent node functions for the Due Diligence Report Generator.

Each function is a LangGraph node that receives state and returns
state updates. Functions are grouped by graph level:
- Orchestrator nodes: planner, synthesizer, final_evaluator
- Analyst nodes: research, write_section, build_completed_section
- Eval-optimize nodes: evaluate_section, refine_section
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from eval_config import get_eval_config
from prompts import (
    ANALYST_PROMPTS,
    FINAL_EVALUATOR_PROMPT,
    PLANNER_PROMPT,
    SECTION_EVALUATOR_PROMPT,
    SECTION_REFINER_PROMPT,
    SYNTHESIZER_PROMPT,
)
from state import (
    AnalysisSection,
    AnalystState,
    CompletedSection,
    EvalOptimizeState,
    EvaluationResult,
    OrchestratorState,
    ResearchData,
    WorkerInput,
)
from tools import TOOLS_BY_ANALYST


def _get_llm(
    *,
    role: str = "default",
) -> Any:
    """Create an LLM instance, spreading load across endpoints and models.

    Args:
        role: Which logical role needs the LLM.  Used to spread calls
            across two Azure OpenAI endpoints and two model deployments
            so the S0 token-rate limits are not exhausted.

            Role → (endpoint, model) mapping:
              - "planner"     → SW endpoint, gpt-5.2 (thinking model)
              - "writer"      → original endpoint, gpt-4.1
              - "evaluator"   → SW endpoint, gpt-4.1
              - "refiner"     → original endpoint, gpt-4.1
              - "synthesizer" → SW endpoint, gpt-5.2
              - "final_eval"  → original endpoint, gpt-4.1
              - "default"     → SW endpoint, gpt-4.1
    """
    from langchain_openai import AzureChatOpenAI

    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Sweden-Central (SW) endpoint — has both gpt-4.1 and gpt-5.2
    sw_endpoint = os.environ.get("AZURE_OPENAI_SW_ENDPOINT", "").rstrip("/")
    sw_key = os.environ.get("AZURE_OPENAI_SW_KEY", "")
    # Strip /openai/v1 suffix if present — AzureChatOpenAI wants the base URL
    if sw_endpoint.endswith("/openai/v1"):
        sw_endpoint = sw_endpoint[: -len("/openai/v1")]

    # Original endpoint
    orig_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    orig_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

    thinking_model = os.environ.get("AZURE_OPENAI_THINKING_MODEL_DEPLOYMENT_NAME", "gpt-5.2")
    chat_model = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1")

    # Map roles → (endpoint, key, deployment)
    # All on SW endpoint, alternating between gpt-4.1 and gpt-5.2
    role_map: dict[str, tuple[str, str, str]] = {
        "planner":     (sw_endpoint, sw_key, thinking_model),   # gpt-5.2
        "writer":      (sw_endpoint, sw_key, chat_model),       # gpt-4.1
        "evaluator":   (sw_endpoint, sw_key, thinking_model),   # gpt-5.2
        "refiner":     (sw_endpoint, sw_key, chat_model),       # gpt-4.1
        "synthesizer": (sw_endpoint, sw_key, thinking_model),   # gpt-5.2
        "final_eval":  (sw_endpoint, sw_key, chat_model),       # gpt-4.1
        "default":     (sw_endpoint, sw_key, chat_model),
    }

    endpoint, key, deployment = role_map.get(role, role_map["default"])

    # Fallback if SW creds missing
    if not endpoint or not key:
        endpoint, key, deployment = orig_endpoint, orig_key, chat_model

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=api_version,
        azure_deployment=deployment,
        temperature=0.3,
        max_retries=12,
    )


def _resolve_company_name(state: AnalystState | WorkerInput | dict[str, Any]) -> str:
    """Resolve company name from worker state, falling back to a safe sample value."""
    return str(state.get("company_name", "TargetCo"))


def _resolve_industry(state: AnalystState | WorkerInput | dict[str, Any]) -> str:
    """Resolve industry from worker state, falling back to the section area."""
    section = state.get("section")
    if "industry" in state:
        return str(state["industry"])
    if isinstance(section, AnalysisSection):
        return section.area
    return "unknown"


def _message_content_to_text(content: Any) -> str:
    """Normalize LangChain message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


# ============================================================
# Structured output schemas
# ============================================================


class PlannedSection(BaseModel):
    """A planned section of the report."""

    area: Literal["financial", "market", "risk", "esg"]
    title: str = Field(description="Title for this report section")
    description: str = Field(description="Detailed description of what to analyze")


class AnalysisPlan(BaseModel):
    """Structured output for the planner."""

    sections: list[PlannedSection]


class SectionEvaluation(BaseModel):
    """Structured output for section evaluation."""

    score: int = Field(ge=1, le=5, description="Quality score from 1 to 5")
    passed: bool = Field(description="Whether the section meets professional standards")
    feedback: str = Field(description="Specific feedback for improvement if not passed")
    strengths: list[str] = Field(
        default_factory=list,
        description="What the section does well",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Areas needing improvement",
    )


class FinalEvaluation(BaseModel):
    """Structured output for final report evaluation."""

    score: int = Field(ge=1, le=5, description="Overall quality score from 1 to 5")
    passed: bool = Field(
        description="Whether the report is ready for the investment committee"
    )
    feedback: str = Field(description="Overall assessment and any issues")


# ============================================================
# Orchestrator nodes
# ============================================================


def planner(state: OrchestratorState) -> dict[str, Any]:
    """Decompose due diligence into specialist analysis areas."""
    llm = _get_llm(role="planner")
    structured_llm = llm.with_structured_output(AnalysisPlan)

    result = structured_llm.invoke(
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(
                content=(
                    f"Plan a due diligence report for {state['company_name']} "
                    f"in the {state['industry']} industry."
                )
            ),
        ]
    )

    sections = [
        AnalysisSection(
            area=section.area,
            title=section.title,
            description=section.description,
            analyst_type=section.area,
        )
        for section in result.sections
    ]

    return {"analysis_plan": sections}


def synthesizer(state: OrchestratorState) -> dict[str, Any]:
    """Combine all completed sections into a final report."""
    llm = _get_llm(role="synthesizer")

    sections_text = "\n\n---\n\n".join(
        f"## {section.title}\n\n{section.content}"
        for section in state["completed_sections"]
    )

    prompt = SYNTHESIZER_PROMPT.format(
        company_name=state["company_name"],
        industry=state["industry"],
        sections=sections_text,
    )

    response = llm.invoke(
        [
            SystemMessage(content="You are a senior investment partner."),
            HumanMessage(content=prompt),
        ]
    )

    return {"final_report": _message_content_to_text(response.content)}


def final_evaluator(state: OrchestratorState) -> dict[str, Any]:
    """Quality gate: evaluate the complete report."""
    llm = _get_llm(role="final_eval")
    structured_llm = llm.with_structured_output(FinalEvaluation)
    final_config = get_eval_config("final")

    prompt = FINAL_EVALUATOR_PROMPT.format(
        company_name=state["company_name"],
        industry=state["industry"],
        report=state["final_report"],
    )

    result = structured_llm.invoke(
        [
            SystemMessage(content="You are a managing director reviewing due diligence."),
            HumanMessage(content=prompt),
        ]
    )

    passed = result.passed and result.score >= max(
        evaluator.pass_threshold for evaluator in final_config.evaluators
    )

    return {
        "final_evaluation": EvaluationResult(
            evaluator_name="final_quality_gate",
            passed=passed,
            score=float(result.score),
            label="pass" if passed else "fail",
            explanation=result.feedback,
        )
    }


# ============================================================
# Analyst (child) nodes
# ============================================================


def research(state: AnalystState) -> dict[str, Any]:
    """Analyst researches using domain-specific tools."""
    section = state["section"]
    company_name = _resolve_company_name(state)
    industry = _resolve_industry(state)
    tools = TOOLS_BY_ANALYST.get(section.analyst_type, [])

    research_data: list[ResearchData] = []
    for tool_fn in tools:
        try:
            args: dict[str, Any] = {}
            args_schema = getattr(tool_fn, "args_schema", None)
            tool_params = (
                cast(dict[str, Any], args_schema.model_json_schema()).get(
                    "properties", {}
                )
                if args_schema
                else {}
            )

            if "company_name" in tool_params:
                args["company_name"] = company_name
            if "industry" in tool_params:
                args["industry"] = industry
            if "years" in tool_params:
                args["years"] = 3

            result = tool_fn.invoke(args)
            research_data.append(
                ResearchData(
                    source=tool_fn.name,
                    data=result if isinstance(result, dict) else {"result": result},
                    summary=f"Data from {tool_fn.name}",
                )
            )
        except Exception as exc:
            research_data.append(
                ResearchData(
                    source=tool_fn.name,
                    data={"error": str(exc)},
                    summary=f"Error from {tool_fn.name}: {exc}",
                )
            )

    return {"research_data": research_data}


def write_section(state: AnalystState) -> dict[str, Any]:
    """Analyst writes a section draft from research data."""
    llm = _get_llm(role="writer")
    section = state["section"]
    company_name = _resolve_company_name(state)
    industry = _resolve_industry(state)

    research_context = "\n\n".join(
        f"### Data from {item.source}:\n```json\n{json.dumps(item.data, indent=2)}\n```"
        for item in state.get("research_data", [])
    )

    prompt_template = ANALYST_PROMPTS.get(
        section.analyst_type, ANALYST_PROMPTS["financial"]
    )
    system_prompt = prompt_template.format(
        company_name=company_name,
        industry=industry,
        section_description=section.description,
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Here is the research data to base your analysis on:\n\n"
                    f"{research_context}\n\n"
                    "Write the section now. Be thorough and cite specific data points."
                )
            ),
        ]
    )

    return {
        "draft_content": _message_content_to_text(response.content),
        "iteration_count": state.get("iteration_count", 0),
    }


def build_completed_section(state: AnalystState) -> dict[str, Any]:
    """Package the analyst's work into a CompletedSection."""
    section = state["section"]
    return {
        "completed_section": CompletedSection(
            area=section.area,
            title=section.title,
            content=state["draft_content"],
            research_data=state.get("research_data", []),
            evaluation_results=(
                [state["evaluation_result"]] if state.get("evaluation_result") else []
            ),
            iterations=state.get("iteration_count", 0),
        )
    }


# ============================================================
# Eval-Optimize (grandchild) nodes
# ============================================================


def evaluate_section(state: EvalOptimizeState) -> dict[str, Any]:
    """Evaluate a section draft using Azure Foundry builtin evaluators.

    Calls the real Foundry evaluator API (e.g., builtin.task_completion)
    via the FoundryEvaluator wrapper, then maps the pass/fail result
    back into the graph state. Falls back to LLM-as-judge when the
    Foundry project endpoint is not configured.
    """
    eval_cfg = get_eval_config(state["section_area"])
    project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    deployment_name = os.environ.get(
        "AZURE_AI_MODEL_DEPLOYMENT_NAME",
        os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o"),
    )

    # --- Try real Foundry evaluator ---
    if project_endpoint and eval_cfg.evaluators:
        try:
            return _evaluate_with_foundry(state, eval_cfg, project_endpoint, deployment_name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Foundry evaluator failed, falling back to LLM-as-judge: %s", e
            )

    # --- Fallback: LLM-as-judge ---
    return _evaluate_with_llm(state, eval_cfg)


def _evaluate_with_foundry(
    state: EvalOptimizeState,
    eval_cfg: Any,
    project_endpoint: str,
    deployment_name: str,
) -> dict[str, Any]:
    """Call the real Foundry builtin evaluator and map results to state."""
    from langchain_azure_ai.evaluation import FoundryEvaluator

    evaluator_config = eval_cfg.evaluators[0]

    foundry_eval = FoundryEvaluator(
        project_endpoint=project_endpoint,
        evaluator_name=evaluator_config.evaluator_name,
        deployment_name=deployment_name,
        display_name=evaluator_config.name,
        poll_interval=3.0,
        max_wait=120.0,
    )

    # Build query/response in Foundry's expected format.
    # query = the assignment (system prompt + user request)
    # response = the analyst's draft output
    query = [
        {"role": "system", "content": f"You are a {state['section_area']} analyst writing a due diligence report section."},
        {"role": "user", "content": [{"type": "text", "text": f"Write a comprehensive {state['section_area']} analysis section titled '{state['section_title']}'."}]},
    ]
    response = [
        {"role": "assistant", "content": [{"type": "text", "text": state["draft_content"]}]},
    ]

    foundry_result = foundry_eval.evaluate(
        query=query,
        response=response,
    )

    # Map Foundry result → graph state
    passed = foundry_result.passed
    feedback = foundry_result.explanation or ""
    if not passed and not feedback:
        feedback = f"Foundry {evaluator_config.name} evaluator returned: {foundry_result.label}"

    return {
        "evaluation_result": EvaluationResult(
            evaluator_name=evaluator_config.evaluator_name,
            passed=passed,
            score=foundry_result.score,
            label=foundry_result.label or ("pass" if passed else "fail"),
            explanation=feedback,
            iteration=state["iteration"],
        ),
        "evaluation_feedback": "" if passed else feedback,
        "accepted": passed,
        "iteration": state["iteration"] + 1,
        "max_iterations": state.get("max_iterations")
        or evaluator_config.max_iterations,
    }


def _evaluate_with_llm(
    state: EvalOptimizeState,
    eval_cfg: Any,
) -> dict[str, Any]:
    """Fallback: evaluate using our own LLM-as-judge."""
    llm = _get_llm(role="evaluator")
    structured_llm = llm.with_structured_output(SectionEvaluation)
    threshold = max(e.pass_threshold for e in eval_cfg.evaluators) if eval_cfg.evaluators else 4.0

    prompt = SECTION_EVALUATOR_PROMPT.format(
        section_description=state["section_title"],
        section_area=state["section_area"],
        draft_content=state["draft_content"],
    )

    result = structured_llm.invoke(
        [
            SystemMessage(content="You are a senior editor evaluating report quality."),
            HumanMessage(content=prompt),
        ]
    )

    passed = result.passed and result.score >= threshold

    return {
        "evaluation_result": EvaluationResult(
            evaluator_name=f"{state['section_area']}_quality",
            passed=passed,
            score=float(result.score),
            label="pass" if passed else "fail",
            explanation=result.feedback,
            iteration=state["iteration"],
        ),
        "evaluation_feedback": "" if passed else result.feedback,
        "accepted": passed,
        "iteration": state["iteration"] + 1,
        "max_iterations": state.get("max_iterations")
        or max(e.max_iterations for e in eval_cfg.evaluators),
    }


def refine_section(state: EvalOptimizeState) -> dict[str, Any]:
    """Refine a section draft based on evaluation feedback."""
    llm = _get_llm(role="refiner")

    prompt = SECTION_REFINER_PROMPT.format(
        section_description=state["section_title"],
        company_name="TargetCo",
        industry=state["section_area"],
        draft_content=state["draft_content"],
        feedback=state["evaluation_feedback"],
    )

    response = llm.invoke(
        [
            SystemMessage(content="You are revising a due diligence report section."),
            HumanMessage(content=prompt),
        ]
    )

    return {"draft_content": _message_content_to_text(response.content)}


# ============================================================
# Routing functions
# ============================================================


def should_refine(state: EvalOptimizeState) -> str:
    """Route: accept the section or send back for refinement."""
    max_iterations = state.get("max_iterations") or max(
        evaluator.max_iterations
        for evaluator in get_eval_config(state["section_area"]).evaluators
    )
    if state["accepted"]:
        return "accepted"
    if state["iteration"] >= max_iterations:
        return "accepted"
    return "refine"


__all__ = [
    "AnalysisPlan",
    "FinalEvaluation",
    "PlannedSection",
    "SectionEvaluation",
    "build_completed_section",
    "evaluate_section",
    "final_evaluator",
    "planner",
    "refine_section",
    "research",
    "should_refine",
    "synthesizer",
    "write_section",
]
