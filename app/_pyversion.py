"""Fail-fast guard against unsupported Python versions.

Single source of truth for supported-version policy at runtime. Imported
from every entrypoint that touches the openai/httpx/anyio code path
affected by the Python 3.14 asyncio cleanup regression. The upper bound
matches pyproject.toml's `requires-python`.
"""
import sys

MIN_VERSION = (3, 12)
MAX_EXCLUSIVE = (3, 13)  # mirrors pyproject.toml's <3.13 (verified-only policy)


def check_python_version() -> None:
    v = sys.version_info[:2]
    if v < MIN_VERSION or v >= MAX_EXCLUSIVE:
        raise RuntimeError(
            f"Unsupported Python version {sys.version}. "
            f"This project requires {MIN_VERSION[0]}.{MIN_VERSION[1]}.x "
            f"(verified-only policy: <{MAX_EXCLUSIVE[0]}.{MAX_EXCLUSIVE[1]}). "
            f"3.13 is unverified; 3.14 has a known anyio/asyncio cleanup regression "
            f"that masks errors from openai requests."
        )


check_python_version()  # run on import
