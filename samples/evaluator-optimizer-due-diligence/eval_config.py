"""Foundry evaluator configurations for the Due Diligence Report Generator.

Defines testing criteria and data source configs for Azure Foundry
agent evaluators. Each analyst type uses a different evaluator to
showcase the range of available Foundry eval capabilities.

Evaluator mapping:
- Financial Analyst → TaskCompletion (did the analyst complete the full analysis?)
- Market Analyst → TaskAdherence (did the analyst follow the systematic methodology?)
- Risk Analyst → TaskCompletion (did the analyst identify all risk categories?)
- ESG Analyst → TaskAdherence (did the analyst cover E, S, and G dimensions?)
- Final Quality Gate → TaskCompletion + TaskAdherence (full report assessment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluatorConfig:
    """Configuration for a single Foundry evaluator."""

    name: str
    evaluator_name: str  # e.g., "builtin.task_completion"
    description: str
    pass_threshold: float = 3.0  # Score threshold for pass/fail
    max_iterations: int = 3  # Max eval-optimize iterations


@dataclass
class AnalystEvalConfig:
    """Complete evaluation configuration for an analyst type."""

    analyst_type: str
    evaluators: list[EvaluatorConfig] = field(default_factory=list)


# --- Per-analyst evaluator configs ---

FINANCIAL_EVAL_CONFIG = AnalystEvalConfig(
    analyst_type="financial",
    evaluators=[
        EvaluatorConfig(
            name="task_completion",
            evaluator_name="builtin.task_completion",
            description=(
                "Evaluates whether the financial analyst completed all required "
                "analysis areas: revenue, profitability, cash flow, balance "
                "sheet, and valuation."
            ),
            pass_threshold=3.0,
            max_iterations=1,
        ),
    ],
)

MARKET_EVAL_CONFIG = AnalystEvalConfig(
    analyst_type="market",
    evaluators=[
        EvaluatorConfig(
            name="task_adherence",
            evaluator_name="builtin.task_adherence",
            description=(
                "Evaluates whether the market analyst adhered to the systematic "
                "methodology: TAM analysis, competitive mapping, trend "
                "identification, and moat assessment."
            ),
            pass_threshold=3.0,
            max_iterations=1,
        ),
    ],
)

RISK_EVAL_CONFIG = AnalystEvalConfig(
    analyst_type="risk",
    evaluators=[
        EvaluatorConfig(
            name="task_completion",
            evaluator_name="builtin.task_completion",
            description=(
                "Evaluates whether the risk analyst identified and assessed all "
                "required risk categories: regulatory, operational, financial, "
                "market, and litigation."
            ),
            pass_threshold=3.0,
            max_iterations=1,
        ),
    ],
)

ESG_EVAL_CONFIG = AnalystEvalConfig(
    analyst_type="esg",
    evaluators=[
        EvaluatorConfig(
            name="task_adherence",
            evaluator_name="builtin.task_adherence",
            description=(
                "Evaluates whether the ESG analyst systematically covered all "
                "three dimensions: Environmental, Social, and Governance."
            ),
            pass_threshold=3.0,
            max_iterations=1,
        ),
    ],
)

FINAL_QUALITY_GATE_CONFIG = AnalystEvalConfig(
    analyst_type="final",
    evaluators=[
        EvaluatorConfig(
            name="task_completion",
            evaluator_name="builtin.task_completion",
            description=(
                "Evaluates whether the complete due diligence report covers all "
                "required sections and provides a clear investment recommendation."
            ),
            pass_threshold=3.0,
            max_iterations=1,
        ),
        EvaluatorConfig(
            name="task_adherence",
            evaluator_name="builtin.task_adherence",
            description=(
                "Evaluates whether the report adheres to professional due "
                "diligence standards and institutional investor expectations."
            ),
            pass_threshold=3.0,
            max_iterations=1,
        ),
    ],
)


# --- Lookup by analyst type ---

EVAL_CONFIGS: dict[str, AnalystEvalConfig] = {
    "financial": FINANCIAL_EVAL_CONFIG,
    "market": MARKET_EVAL_CONFIG,
    "risk": RISK_EVAL_CONFIG,
    "esg": ESG_EVAL_CONFIG,
    "final": FINAL_QUALITY_GATE_CONFIG,
}


def get_eval_config(analyst_type: str) -> AnalystEvalConfig:
    """Get evaluation configuration for an analyst type.

    Args:
        analyst_type: One of 'financial', 'market', 'risk', 'esg', 'final'.

    Returns:
        The evaluation configuration for the specified analyst type.

    Raises:
        ValueError: If analyst_type is not recognized.
    """
    if analyst_type not in EVAL_CONFIGS:
        raise ValueError(
            f"Unknown analyst type: {analyst_type!r}. "
            f"Available: {list(EVAL_CONFIGS.keys())}"
        )
    return EVAL_CONFIGS[analyst_type]


def build_testing_criteria(
    config: AnalystEvalConfig,
    deployment_name: str,
) -> list[dict[str, Any]]:
    """Build Foundry testing_criteria from an analyst eval config.

    Args:
        config: The analyst evaluation configuration.
        deployment_name: The Azure AI model deployment name for the LLM judge.

    Returns:
        List of testing criteria dicts suitable for client.evals.create().
    """
    criteria = []
    for evaluator in config.evaluators:
        criteria.append(
            {
                "type": "azure_ai_evaluator",
                "name": evaluator.name,
                "evaluator_name": evaluator.evaluator_name,
                "initialization_parameters": {
                    "deployment_name": deployment_name,
                },
                "data_mapping": {
                    "query": "{{item.query}}",
                    "response": "{{item.response}}",
                },
            }
        )
    return criteria


def build_data_source_config() -> dict[str, Any]:
    """Build the data source config schema for Foundry evals.

    Returns:
        Data source config dict suitable for client.evals.create().
    """
    return {
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "object"}},
                    ]
                },
                "response": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "object"}},
                    ]
                },
                "tool_definitions": {
                    "anyOf": [
                        {"type": "object"},
                        {"type": "array", "items": {"type": "object"}},
                    ]
                },
            },
            "required": ["query", "response"],
        },
        "include_sample_schema": True,
    }
