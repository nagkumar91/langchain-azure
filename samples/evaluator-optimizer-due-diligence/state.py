"""State schemas for the Due Diligence Report Generator.

Defines TypedDict schemas for the 3-level LangGraph hierarchy:
- OrchestratorState (parent): Top-level report coordination
- AnalystState (child): Individual analyst worker state
- EvalOptimizeState (grandchild): Evaluator-optimizer loop state
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


# --- Data models ---


@dataclass
class AnalysisSection:
    """A section of the due diligence report."""

    area: str  # e.g., "financial", "market", "risk", "esg"
    title: str
    description: str  # What the analyst should cover
    analyst_type: Literal["financial", "market", "risk", "esg"]


@dataclass
class ResearchData:
    """Research data gathered by an analyst."""

    source: str
    data: dict[str, Any]
    summary: str


@dataclass
class EvaluationResult:
    """Result from a Foundry evaluation."""

    evaluator_name: str
    passed: bool
    score: float | None = None
    label: str | None = None  # "pass" or "fail"
    explanation: str | None = None
    iteration: int = 0


@dataclass
class CompletedSection:
    """A completed section of the report."""

    area: str
    title: str
    content: str
    research_data: list[ResearchData] = field(default_factory=list)
    evaluation_results: list[EvaluationResult] = field(default_factory=list)
    iterations: int = 0


# --- Graph states (3 levels) ---


class OrchestratorState(TypedDict):
    """Parent graph state: coordinates the entire due diligence report."""

    company_name: str
    industry: str
    analysis_plan: list[AnalysisSection]  # Planner output
    completed_sections: Annotated[list[CompletedSection], operator.add]  # Workers write here
    final_report: str
    final_evaluation: EvaluationResult | None


class AnalystState(TypedDict):
    """Child graph state: individual analyst researches and writes a section.

    Note: Different state schema from parent — connected via wrapper node.
    """

    section: AnalysisSection  # Assignment from orchestrator
    research_data: list[ResearchData]
    draft_content: str
    evaluation_result: EvaluationResult | None
    completed_section: CompletedSection | None
    iteration_count: int


class EvalOptimizeState(TypedDict):
    """Grandchild graph state: evaluator-optimizer feedback loop.

    Note: Different state schema from child — connected via wrapper node.
    """

    section_area: str
    section_title: str
    draft_content: str
    evaluation_feedback: str
    evaluation_result: EvaluationResult | None
    accepted: bool
    iteration: int
    max_iterations: int


# --- Worker state for Send API ---


class WorkerInput(TypedDict):
    """Input dispatched to each analyst worker via Send API."""

    section: AnalysisSection
    company_name: str
    industry: str
