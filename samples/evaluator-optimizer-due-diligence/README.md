# Evaluator-Optimizer Due Diligence Report Generator

A complex, long-running, fully autonomous multi-agent LangGraph sample that generates
investment due diligence reports using Azure Foundry evaluators for quality assurance.

## Architecture

This sample demonstrates a **3-level LangGraph hierarchy** with the evaluator-optimizer pattern:

```
Orchestrator (Parent Graph — OrchestratorState)
├── Planner Node → decomposes analysis into 4 specialist areas
├── Send API → spawns parallel analyst workers
│   ├── Financial Analyst (Child Subgraph — AnalystState)
│   │   ├── Research → gathers financial data (mock tools)
│   │   ├── Writer → drafts financial analysis section
│   │   └── Eval-Optimize Loop (Grandchild Subgraph — EvalOptimizeState)
│   │       ├── Evaluator → scores quality with structured output
│   │       └── Conditional → refine or accept (max 3 iterations)
│   ├── Market Analyst (same pattern, TaskAdherence evaluator)
│   ├── Risk Analyst (same pattern, TaskCompletion evaluator)
│   └── ESG Analyst (same pattern, TaskAdherence evaluator)
├── Synthesizer → combines all sections into final report
└── Final Quality Gate → Foundry TaskCompletion + TaskAdherence
```

### Key LangGraph Patterns Demonstrated

| Pattern | Where Used |
|---------|-----------|
| **Send API** | Orchestrator → parallel analyst workers |
| **Subgraphs (different state schemas)** | Parent→Child→Grandchild, each with own TypedDict |
| **Evaluator-Optimizer Loop** | Grandchild graph: evaluate → conditional refine |
| **Conditional Routing** | Eval loop exit condition (pass/max iterations) |
| **State Transformation** | Wrapper nodes transform state between levels |
| **Parallel Execution** | All 4 analyst workers run concurrently |

### Foundry Evaluator Integration

Each analyst uses a different Azure Foundry evaluator to showcase variety:

| Analyst | Foundry Evaluator | Purpose |
|---------|------------------|---------|
| Financial | `builtin.task_completion` | Did the analyst complete all analysis areas? |
| Market | `builtin.task_adherence` | Did the analyst follow systematic methodology? |
| Risk | `builtin.task_completion` | Did the analyst identify all risk categories? |
| ESG | `builtin.task_adherence` | Did the analyst cover E, S, G dimensions? |
| Final Gate | Both evaluators | Full report quality assessment |

## Prerequisites

- Python 3.10+
- Azure AI Foundry project (for Foundry evaluators)
- Azure OpenAI or OpenAI API key (for LLM)

## Setup

```bash
# Install dependencies
pip install langchain-azure-ai[opentelemetry] langgraph python-dotenv langchain-openai

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_AI_PROJECT_ENDPOINT` | Yes | Azure AI Foundry project endpoint |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Yes | Model deployment name (e.g., `gpt-4o`) |
| `APPLICATION_INSIGHTS_CONNECTION_STRING` | No | For OpenTelemetry tracing |
| `TARGET_COMPANY` | No | Default company name |
| `TARGET_INDUSTRY` | No | Default industry sector |

## Usage

```bash
# Basic run
python main.py

# Custom company
python main.py --company "Tesla" --industry "Electric Vehicles"

# With tracing
python main.py --trace

# Save report to file
python main.py --output report.md
```

## File Structure

```
evaluator-optimizer-due-diligence/
├── main.py          # Entry point with CLI
├── graph.py         # 3-level LangGraph hierarchy
├── agents.py        # Agent node functions (planner, analysts, evaluators)
├── state.py         # TypedDict state schemas for all 3 levels
├── prompts.py       # System prompts for all agents
├── tools.py         # Mock research tools (12 tools across 4 domains)
├── eval_config.py   # Foundry evaluator configurations
├── README.md        # This file
├── .env.example     # Environment variable template
└── requirements.txt # Python dependencies
```

## How It Works

1. **Planning**: The orchestrator's planner decomposes the due diligence into 4 analysis areas
2. **Parallel Research**: Each specialist analyst runs concurrently via the Send API
3. **Tool Usage**: Analysts use domain-specific mock tools to gather research data
4. **Writing**: Each analyst drafts a comprehensive section from research data
5. **Evaluation Loop**: Each draft enters an evaluator-optimizer loop:
   - LLM-as-judge evaluates quality with structured output
   - If score < 4/5, feedback is generated and draft is refined
   - Loop continues until accepted or max iterations (3) reached
6. **Synthesis**: All completed sections are synthesized into a final report
7. **Quality Gate**: The complete report undergoes final evaluation

## Library Integration

This sample uses the `langchain_azure_ai.evaluation` module which provides:

```python
from langchain_azure_ai.evaluation import (
    FoundryEvaluator,          # Single evaluator wrapper
    FoundryEvaluatorSuite,     # Multiple evaluators
    FoundryEvalResult,         # Structured result
    messages_to_foundry_format, # LangChain → Foundry converter
    create_eval_optimize_subgraph,  # Graph builder helper
)
```

The tracer emits `gen_ai.evaluation.result` OpenTelemetry events per the
[GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/model/gen-ai/events.yaml).
