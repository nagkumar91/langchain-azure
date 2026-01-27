"""Shared pytest configuration for integration tests with HTTP recordings."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, cast

import pytest
from vcr import VCR  # type: ignore[import-not-found, import-untyped]

FILTER_HEADERS = [
    ("authorization", "REDACTED"),
    ("api-key", "REDACTED"),
    ("x-api-key", "REDACTED"),
    ("openai-api-key", "REDACTED"),
    ("openai-organization", "REDACTED"),
    ("user-agent", "REDACTED"),
    ("x-openai-client-user-agent", "REDACTED"),
]


def _remove_header_case_insensitive(
    headers: MutableMapping[str, Any],
    header_name: str,
) -> None:
    """Remove headers that match header_name (case-insensitive)."""
    match_name = header_name.lower()
    keys_to_remove = [key for key in headers if key.lower() == match_name]
    for key in keys_to_remove:
        del headers[key]


def _sanitize_request(request: Any) -> Any:
    headers = cast(MutableMapping[str, Any], getattr(request, "headers", {}))
    _remove_header_case_insensitive(headers, "cookie")
    for header, replacement in FILTER_HEADERS:
        if header in headers:
            headers[header] = replacement
    return request


def _sanitize_response(response: Dict[str, Any]) -> Dict[str, Any]:
    headers = cast(MutableMapping[str, Any], response.get("headers", {}))
    for header in list(headers):
        if header.lower() != "set-cookie":
            headers[header] = ["REDACTED"]
    _remove_header_case_insensitive(headers, "set-cookie")
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Base configuration for pytest-recording/VCR."""
    return {
        "cassette_library_dir": "tests/integration_tests/cassettes",
        "filter_headers": FILTER_HEADERS,
        "match_on": ["method", "uri", "body"],
        "decode_compressed_response": True,
        "before_record_request": _sanitize_request,
        "before_record_response": _sanitize_response,
        "path_transformer": VCR.ensure_suffix(".yaml"),
    }
