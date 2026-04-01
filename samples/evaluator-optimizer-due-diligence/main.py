"""Autonomous Due Diligence Report Generator — Entry Point.

A long-running, fully autonomous multi-agent system that generates
comprehensive investment due diligence reports using LangGraph with
Azure Foundry evaluators for quality assurance.

Architecture (3-level hierarchy):
    Orchestrator → Specialist Analysts (×4) → Eval-Optimize Loops

Usage:
    # Set environment variables (see .env.example)
    python main.py

    # With custom company
    python main.py --company "Acme Corp" --industry "Technology"

    # With tracing enabled
    python main.py --trace
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

# Add parent path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load the sample's own .env first, then any override .env
load_dotenv()
# Also try the reference local_samples .env for shared credentials
_ref_env = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "libs", "azure-ai", "samples", "local_samples", ".env",
)
if os.path.exists(_ref_env):
    load_dotenv(_ref_env, override=False)

# Enable content recording for Azure tracing
os.environ.setdefault("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Quiet down verbose Azure HTTP logging — keep our app + key SDK messages
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.monitor.opentelemetry.exporter").setLevel(logging.WARNING)
logging.getLogger("opentelemetry.attributes").setLevel(logging.WARNING)
LOGGER = logging.getLogger("due_diligence")


def setup_tracing(enable: bool = False) -> Any:
    """Configure OpenTelemetry tracing if enabled.

    Returns:
        The tracer callback handler, or None if tracing is disabled.
    """
    if not enable:
        return None

    try:
        from langchain_azure_ai.callbacks.tracers.inference_tracing import (
            AzureAIOpenTelemetryTracer,
        )

        tracer = AzureAIOpenTelemetryTracer(
            connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
            project_endpoint=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
            enable_content_recording=True,
            name="DueDiligenceReportGenerator",
        )
        LOGGER.info("OpenTelemetry tracing enabled")
        return tracer
    except ImportError:
        LOGGER.warning(
            "Tracing requested but azure-monitor-opentelemetry not installed. "
            "Install with: pip install azure-monitor-opentelemetry"
        )
        return None
    except Exception as e:
        LOGGER.warning("Failed to initialize tracing: %s", e)
        return None


def run_due_diligence(
    company_name: str,
    industry: str,
    *,
    tracer: Any = None,
    verbose: bool = True,
    parallel: bool = False,
) -> dict[str, Any]:
    """Run the due diligence report generator.

    Args:
        company_name: Target company name.
        industry: Industry sector.
        tracer: Optional OTel tracer callback handler.
        verbose: Print progress to stdout.
        parallel: Use parallel Send API (True) or sequential (False).

    Returns:
        The final orchestrator state with the complete report.
    """
    from graph import create_due_diligence_graph
    from state import OrchestratorState

    mode = "parallel (Send API)" if parallel else "sequential"

    if verbose:
        print(f"\n{'='*70}")
        print("  Due Diligence Report Generator")
        print(f"  Company: {company_name}")
        print(f"  Industry: {industry}")
        print(f"  Mode: {mode}")
        print(f"{'='*70}\n")

    # Build the graph
    graph = create_due_diligence_graph(parallel=parallel)

    if verbose:
        nodes = list(graph.get_graph().nodes.keys())
        print(f"📊 Graph compiled with nodes: {nodes}")
        print("🚀 Starting autonomous analysis...\n")

    # Prepare initial state
    initial_state: OrchestratorState = {
        "company_name": company_name,
        "industry": industry,
        "analysis_plan": [],
        "completed_sections": [],
        "final_report": "",
        "final_evaluation": None,
    }

    # Configure callbacks and metadata for tracing
    config: dict[str, Any] = {
        "metadata": {
            "agent_name": "DueDiligenceOrchestrator",
            "agent_type": "orchestrator",
        },
    }
    if tracer:
        config["callbacks"] = [tracer]

    # Run the graph — single invoke to avoid double-execution
    start_time = time.time()

    if verbose:
        print("⏳ Running full pipeline (planner → analysts → eval loops → synthesizer)...")
        print("   This is a long-running autonomous workflow with multiple LLM calls.\n")

    result = graph.invoke(initial_state, config=config)

    elapsed = time.time() - start_time

    if verbose:
        _print_report(result, elapsed)

    return result


def _print_progress(chunk: Any) -> None:
    """Print streaming progress updates."""
    try:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            ns, data = chunk
            if isinstance(data, dict):
                for node_name in data:
                    indent = "  " * len(ns) if isinstance(ns, (list, tuple)) else ""
                    print(f"{indent}✅ Completed: {node_name}")
    except Exception:
        pass


def _print_report(state: dict[str, Any], elapsed: float) -> None:
    """Print the final report and evaluation results."""
    print(f"\n{'='*70}")
    print(f"  REPORT COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}\n")

    # Print section summaries
    sections = state.get("completed_sections", [])
    print(f"📋 Sections completed: {len(sections)}")
    for sec in sections:
        area = getattr(sec, "area", "unknown")
        title = getattr(sec, "title", "untitled")
        iterations = getattr(sec, "iterations", 0)
        content_len = len(getattr(sec, "content", ""))
        print(f"  • {title} [{area}] — {content_len} chars, {iterations} eval iterations")

    # Print final evaluation
    eval_result = state.get("final_evaluation")
    if eval_result:
        passed = getattr(eval_result, "passed", False)
        score = getattr(eval_result, "score", None)
        label = getattr(eval_result, "label", "unknown")
        explanation = getattr(eval_result, "explanation", "")
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n🎯 Final Quality Gate: {status}")
        if score:
            print(f"   Score: {score}/5.0 ({label})")
        if explanation:
            print(f"   Assessment: {explanation[:200]}...")

    # Print report excerpt
    report = state.get("final_report", "")
    if report:
        print("\n📄 Report Preview (first 500 chars):")
        print(f"{'─'*50}")
        print(report[:500])
        print(f"{'─'*50}")
        print(f"\n📄 Full report: {len(report)} characters")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Due Diligence Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --company "Tesla" --industry "Electric Vehicles"
  python main.py --trace --verbose
        """,
    )
    parser.add_argument(
        "--company",
        default=os.environ.get("TARGET_COMPANY", "Acme Corp"),
        help="Target company name (default: Acme Corp)",
    )
    parser.add_argument(
        "--industry",
        default=os.environ.get("TARGET_INDUSTRY", "Enterprise Software"),
        help="Industry sector (default: Enterprise Software)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Enable OpenTelemetry tracing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress and report (default: True)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run analysts in parallel via Send API (requires high TPM tier)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save full report to file",
    )

    args = parser.parse_args()

    tracer = setup_tracing(args.trace)

    try:
        result = run_due_diligence(
            company_name=args.company,
            industry=args.industry,
            tracer=tracer,
            verbose=args.verbose,
            parallel=args.parallel,
        )

        if args.output:
            report = result.get("final_report", "")
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\n💾 Report saved to: {args.output}")

    except Exception as e:
        LOGGER.error("Due diligence generation failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
