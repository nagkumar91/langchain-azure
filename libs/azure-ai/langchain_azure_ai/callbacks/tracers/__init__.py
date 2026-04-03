"""Tracing capabilities for Azure AI Foundry."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.callbacks.tracers.auto_instrument import (
        AzureAILangChainInstrumentor,
        disable_auto_tracing,
        enable_auto_tracing,
        is_auto_tracing_enabled,
    )
    from langchain_azure_ai.callbacks.tracers.inference_tracing import (
        AzureAIOpenTelemetryTracer,
    )

__all__ = [
    "AzureAILangChainInstrumentor",
    "AzureAIOpenTelemetryTracer",
    "disable_auto_tracing",
    "enable_auto_tracing",
    "is_auto_tracing_enabled",
]

_module_lookup = {
    "AzureAILangChainInstrumentor": (
        "langchain_azure_ai.callbacks.tracers.auto_instrument"
    ),
    "AzureAIOpenTelemetryTracer": (
        "langchain_azure_ai.callbacks.tracers.inference_tracing"
    ),
    "disable_auto_tracing": "langchain_azure_ai.callbacks.tracers.auto_instrument",
    "enable_auto_tracing": "langchain_azure_ai.callbacks.tracers.auto_instrument",
    "is_auto_tracing_enabled": ("langchain_azure_ai.callbacks.tracers.auto_instrument"),
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
