"""
Langfuse v3 tracing abstraction layer (Phase 5).

Public API:
    init_tracing()          — call at startup; no-op when LANGFUSE_HOST is empty
    trace_span(name, ...)   — context manager; yields _NoopSpan when disabled
    get_callback_handler()  — returns CallbackHandler or None
    flush_tracing()         — call at shutdown

LANGFUSE_HOST empty   → always no-op (intentional dev/test disable)
LANGFUSE_HOST set + connection fails:
    LANGFUSE_REQUIRED=true  → exponential backoff (1→2→4→8→16s), then RuntimeError
    LANGFUSE_REQUIRED=false → warn, tracing disabled (service continues)
"""
import logging
import time
from contextlib import contextmanager
from typing import Any

from settings import settings

logger = logging.getLogger(__name__)

_client: Any = None
_enabled: bool = False

_BACKOFF_DELAYS = (1, 2, 4, 8, 16)


class _NoopSpan:
    """Stub span returned when tracing is disabled."""
    def update(self, **kwargs) -> None:
        pass

    def update_trace(self, **kwargs) -> None:
        pass


def init_tracing() -> None:
    """Initialize Langfuse client at startup.

    Reads LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    LANGFUSE_REQUIRED from settings.  When host is empty, silently
    becomes a no-op.  When host is set but unreachable, retries with
    exponential backoff; on exhaustion raises RuntimeError (required=True)
    or warns and disables tracing (required=False).
    """
    global _client, _enabled

    if not settings.langfuse_host:
        logger.debug("LANGFUSE_HOST not set — tracing disabled")
        return

    from langfuse import Langfuse  # imported lazily to keep import cost low

    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    last_exc: Exception | None = None
    for attempt, delay in enumerate(_BACKOFF_DELAYS):
        try:
            lf.auth_check()
            _client = lf
            _enabled = True
            logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Langfuse auth_check failed (attempt %d/%d): %s",
                attempt + 1, len(_BACKOFF_DELAYS), type(exc).__name__,
            )
            time.sleep(delay)

    # All retries exhausted
    if settings.langfuse_required:
        raise RuntimeError(
            f"Langfuse unreachable after {len(_BACKOFF_DELAYS)} attempts "
            f"(host={settings.langfuse_host}): {type(last_exc).__name__}"
        )
    logger.warning(
        "Langfuse unreachable — LANGFUSE_REQUIRED=false, continuing without tracing"
    )


@contextmanager
def trace_span(
    name: str,
    *,
    input: Any = None,
    metadata: Any = None,
    as_root: bool = False,
    trace_attrs: dict | None = None,
):
    """Context manager that wraps a Langfuse span.

    When tracing is disabled, yields a _NoopSpan that silently ignores
    all update() calls.

    When as_root=True and trace_attrs is provided, calls span.update_trace()
    to attach user_id / session_id / metadata / tags to the root trace.
    """
    if not _enabled or _client is None:
        yield _NoopSpan()
        return

    kwargs: dict = {"name": name}
    if input is not None:
        kwargs["input"] = input
    if metadata is not None:
        kwargs["metadata"] = metadata

    with _client.start_as_current_span(**kwargs) as span:
        if as_root and trace_attrs:
            span.update_trace(**trace_attrs)
        yield span


def get_callback_handler() -> Any | None:
    """Return a LangChain CallbackHandler bound to the current trace context.

    Returns None when tracing is disabled so callers can skip config injection.
    The handler inherits the active OpenTelemetry context, ensuring all LLM
    generations land under the currently open span.
    """
    if not _enabled:
        return None

    from langfuse.langchain import CallbackHandler  # noqa: PLC0415
    return CallbackHandler()


def flush_tracing() -> None:
    """Flush pending Langfuse events at shutdown."""
    if _enabled and _client is not None:
        _client.flush()
        logger.debug("Langfuse flush complete")
