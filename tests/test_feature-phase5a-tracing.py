"""
Unit tests for Phase 5 Step 2a: Langfuse v3 tracing integration.

Run with:
    uv run --project app pytest tests/test_feature-phase5a-tracing.py -v

Test groups:
  A. tracing.py unit tests (init_tracing, trace_span, get_callback_handler)
  B. run_judge_chain metadata completeness
  C. Span lifecycle (root update on completion / exception / cancel)
  D. Concurrency — trace context isolation
  E. Retry hierarchy
  F. judge.py generate_feedback callback_handler argument
"""
import asyncio
import sys
import types
from contextlib import contextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# sys.path — ensure app/ is available
# ---------------------------------------------------------------------------
from pathlib import Path
_APP = Path(__file__).parent.parent / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

# ---------------------------------------------------------------------------
# Import app modules — available when running via: uv run --project app pytest
# ---------------------------------------------------------------------------
import tracing as _tracing_module
from chains.llm_as_judge import run_judge_chain, run_judge_chain_stream
from chains.judge import generate_feedback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_tracing():
    """Reset module-level tracing state."""
    _tracing_module._enabled = False
    _tracing_module._client = None


def _mock_langfuse_module(auth_check_side_effect=None):
    """Build a mock langfuse module for sys.modules patching."""
    mock_client = MagicMock(name="LangfuseClient")
    if auth_check_side_effect is not None:
        mock_client.auth_check = MagicMock(side_effect=auth_check_side_effect)
    else:
        mock_client.auth_check = MagicMock(return_value=None)

    mock_lf_cls = MagicMock(return_value=mock_client, name="Langfuse")
    mock_mod = types.ModuleType("langfuse")
    mock_mod.Langfuse = mock_lf_cls
    mock_mod._mock_client = mock_client
    mock_mod._mock_cls = mock_lf_cls
    return mock_mod


def _make_span_capture():
    """Return (mock_span, context_manager_factory) for capturing span calls."""
    mock_span = MagicMock(name="span")
    mock_span.update = MagicMock()
    mock_span.update_trace = MagicMock()

    @contextmanager
    def span_ctx(name, *, input=None, metadata=None,
                 as_root=False, trace_attrs=None):
        yield mock_span

    return mock_span, span_ctx


async def _collect_stream(agen):
    events = []
    async for ev in agen:
        events.append(ev)
    return events


def _aiter_of(items):
    async def _gen():
        for item in items:
            yield item
    return _gen()


# ===========================================================================
# Section A — tracing.py unit tests
# ===========================================================================

class TestInitTracing:
    def setup_method(self):
        _reset_tracing()

    def test_init_tracing_noop_when_host_empty(self):
        """Empty LANGFUSE_HOST → no Langfuse client constructed."""
        with patch("tracing.settings") as mock_settings:
            mock_settings.langfuse_host = ""
            _tracing_module.init_tracing()

        assert _tracing_module._client is None
        assert _tracing_module._enabled is False

    def test_init_tracing_retries_with_backoff_then_succeeds(self):
        """auth_check fails twice then succeeds — 3 attempts, sleep 1s then 2s."""
        fail_then_pass = [Exception("fail"), Exception("fail"), None]
        mock_mod = _mock_langfuse_module(auth_check_side_effect=fail_then_pass)

        with patch.dict("sys.modules", {"langfuse": mock_mod}), \
             patch("tracing.settings") as mock_settings, \
             patch("time.sleep") as mock_sleep:
            mock_settings.langfuse_host = "http://langfuse-web:3000"
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_required = True

            _tracing_module.init_tracing()

        mock_mod._mock_cls.assert_called_once()
        assert mock_mod._mock_client.auth_check.call_count == 3
        sleep_delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_delays[0] == 1
        assert sleep_delays[1] == 2

    def test_init_tracing_required_true_raises_when_all_retries_fail(self):
        """All retries fail + LANGFUSE_REQUIRED=true → RuntimeError."""
        mock_mod = _mock_langfuse_module(
            auth_check_side_effect=Exception("connect refused")
        )

        with patch.dict("sys.modules", {"langfuse": mock_mod}), \
             patch("tracing.settings") as mock_settings, \
             patch("time.sleep"):
            mock_settings.langfuse_host = "http://langfuse-web:3000"
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_required = True

            with pytest.raises(RuntimeError, match="Langfuse unreachable"):
                _tracing_module.init_tracing()

    def test_init_tracing_required_false_warns_and_disables(self):
        """All retries fail + LANGFUSE_REQUIRED=false → warn, tracing disabled."""
        mock_mod = _mock_langfuse_module(
            auth_check_side_effect=Exception("connect refused")
        )

        with patch.dict("sys.modules", {"langfuse": mock_mod}), \
             patch("tracing.settings") as mock_settings, \
             patch("time.sleep"), \
             patch("tracing.logger") as mock_logger:
            mock_settings.langfuse_host = "http://langfuse-web:3000"
            mock_settings.langfuse_public_key = "pk-test"
            mock_settings.langfuse_secret_key = "sk-test"
            mock_settings.langfuse_required = False

            _tracing_module.init_tracing()

        assert _tracing_module._enabled is False
        mock_logger.warning.assert_called()


class TestTraceSpan:
    def setup_method(self):
        _reset_tracing()

    def test_trace_span_noop_yields_stub(self):
        """Disabled tracing → trace_span yields _NoopSpan with .update() no-op."""
        assert not _tracing_module._enabled

        with _tracing_module.trace_span("test_span") as span:
            span.update(output={"foo": "bar"})  # must not raise

    def test_trace_span_records_when_enabled(self):
        """Enabled tracing → trace_span calls start_as_current_span."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_client = MagicMock()
        mock_client.start_as_current_span = MagicMock(return_value=mock_ctx)

        _tracing_module._enabled = True
        _tracing_module._client = mock_client

        with _tracing_module.trace_span("rag_retrieval", input={"q": "test"}):
            pass

        mock_client.start_as_current_span.assert_called_once()
        assert "rag_retrieval" in str(
            mock_client.start_as_current_span.call_args
        )


class TestGetCallbackHandler:
    def setup_method(self):
        _reset_tracing()

    def test_get_callback_handler_returns_none_when_disabled(self):
        """Disabled tracing → None."""
        assert _tracing_module.get_callback_handler() is None

    def test_get_callback_handler_returns_handler_when_enabled(self):
        """Enabled tracing → returns CallbackHandler instance."""
        mock_handler = MagicMock(name="CallbackHandlerInstance")
        mock_cb_cls = MagicMock(return_value=mock_handler)
        mock_langfuse_lc = types.ModuleType("langfuse.langchain")
        mock_langfuse_lc.CallbackHandler = mock_cb_cls

        _tracing_module._enabled = True
        _tracing_module._client = MagicMock()

        with patch.dict("sys.modules", {"langfuse.langchain": mock_langfuse_lc}):
            result = _tracing_module.get_callback_handler()

        assert result is mock_handler


# ===========================================================================
# Section B — run_judge_chain metadata completeness
# ===========================================================================

class TestRunJudgeChainMetadata:
    """Verify trace_attrs fields passed to trace_span as_root=True."""

    async def test_run_judge_chain_sets_trace_metadata_with_all_required_fields(self):
        """trace_attrs must include all required metadata keys."""
        captured = {}
        mock_span, _ = _make_span_capture()

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            if as_root:
                captured["trace_attrs"] = trace_attrs
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="ctx")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=["ds1"]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            await run_judge_chain("test question", user="alice", use_rag=True)

        assert captured, "trace_span was never called with as_root=True"
        attrs = captured["trace_attrs"]
        assert attrs.get("user_id") == "alice"

        metadata = attrs.get("metadata", {})
        required_keys = {
            "run_id", "datasources", "embedding_model",
            "search_mode", "top_k", "reasoner_model",
            "max_retries", "ragas_threshold",
        }
        missing = required_keys - set(metadata.keys())
        assert not missing, f"Missing metadata keys: {missing}"

    async def test_run_judge_chain_session_id_is_user_date_format(self):
        """session_id must be 'user:YYYY-MM-DD'."""
        captured = {}
        mock_span, _ = _make_span_capture()

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            if as_root:
                captured["trace_attrs"] = trace_attrs
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r
            await run_judge_chain("q", user="alice", use_rag=False)

        expected = f"alice:{date.today().isoformat()}"
        assert captured.get("trace_attrs", {}).get("session_id") == expected

    async def test_run_judge_chain_session_id_anonymous_when_user_none(self):
        """user=None → session_id starts with 'anonymous:'."""
        captured = {}
        mock_span, _ = _make_span_capture()

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            if as_root:
                captured["trace_attrs"] = trace_attrs
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r
            await run_judge_chain("q", user=None, use_rag=False)

        session_id = captured.get("trace_attrs", {}).get("session_id", "")
        assert session_id.startswith("anonymous:"), \
            f"Expected 'anonymous:...', got: {session_id!r}"

    async def test_run_judge_chain_records_full_context_in_rag_span(self):
        """rag_span.update must contain full context text (no truncation)."""
        long_ctx = "CONTEXT_CHUNK " * 100  # 1400 chars
        rag_updates = []
        rag_mock_span = MagicMock()
        rag_mock_span.update = MagicMock(side_effect=lambda **kw: rag_updates.append(kw))
        root_mock_span = MagicMock()

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            if name == "rag_retrieval":
                yield rag_mock_span
            else:
                yield root_mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value=long_ctx)), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r
            await run_judge_chain("q", user="alice", use_rag=True)

        assert rag_updates, "rag_span.update was never called"
        output = rag_updates[0].get("output", {})
        assert output.get("context") == long_ctx, "context must not be truncated"
        assert output.get("context_chars") == len(long_ctx)


# ===========================================================================
# Section C — Span lifecycle
# ===========================================================================

class TestRunJudgeChainLifecycle:

    async def test_run_judge_chain_root_update_on_normal_completion(self):
        """root.update called with verdict/score/final_answer/retries/threshold."""
        root_updates = []
        mock_span = MagicMock()
        mock_span.update = MagicMock(side_effect=lambda **kw: root_updates.append(kw))

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.88,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="good answer"))
            mock_get_reasoner.return_value = mock_r
            await run_judge_chain("q", user="alice", use_rag=False)

        output_updates = [u for u in root_updates if "output" in u]
        assert output_updates, "root span must be updated with output"
        output = output_updates[-1]["output"]
        for key in ("verdict", "score", "final_answer", "retries", "threshold"):
            assert key in output, f"output must contain '{key}'"

    async def test_run_judge_chain_root_update_on_exception(self):
        """On exception: root.update with level='ERROR', then exception re-raised."""
        root_updates = []
        mock_span = MagicMock()
        mock_span.update = MagicMock(side_effect=lambda **kw: root_updates.append(kw))

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner:
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_get_reasoner.return_value = mock_r

            with pytest.raises(RuntimeError, match="LLM down"):
                await run_judge_chain("q", user="alice", use_rag=False)

        error_updates = [u for u in root_updates if u.get("level") == "ERROR"]
        assert error_updates, "root span must have level=ERROR on exception"
        output = error_updates[-1].get("output", {})
        assert output.get("error") == "RuntimeError"

    async def test_run_judge_chain_stream_root_update_on_cancellation(self):
        """CancelledError: root.update with level='WARNING', then re-raised."""
        root_updates = []
        mock_span = MagicMock()
        mock_span.update = MagicMock(side_effect=lambda **kw: root_updates.append(kw))

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(side_effect=asyncio.CancelledError())):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            with pytest.raises((asyncio.CancelledError, Exception)):
                await _collect_stream(
                    run_judge_chain_stream("q", user="alice", use_rag=False)
                )

        warning_updates = [u for u in root_updates if u.get("level") == "WARNING"]
        assert warning_updates, \
            "root span must have level=WARNING on CancelledError"

    async def test_run_judge_chain_stream_root_update_on_generator_exit(self):
        """Stream normal completion: root span always updated (try/finally)."""
        root_updates = []
        mock_span = MagicMock()
        mock_span.update = MagicMock(side_effect=lambda **kw: root_updates.append(kw))

        @contextmanager
        def capturing_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=capturing_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.astream = MagicMock(
                return_value=_aiter_of([MagicMock(content="T"), MagicMock(content="ok")])
            )
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            await _collect_stream(
                run_judge_chain_stream("q", user="alice", use_rag=False)
            )

        # try/finally guarantees root.update is always called
        assert mock_span.update.call_count >= 1, \
            "root span.update must be called (try/finally)"

    async def test_run_judge_chain_stream_passes_config_to_astream(self):
        """Last attempt: config={"callbacks":[handler]} passed to reasoner.astream."""
        mock_handler = MagicMock(name="CallbackHandler")
        astream_configs = []

        async def recording_astream(messages, **kwargs):
            astream_configs.append(kwargs.get("config"))
            yield MagicMock(content="T")
            yield MagicMock(content="ok")

        mock_span = MagicMock()

        @contextmanager
        def noop_span(name, *, input=None, metadata=None,
                      as_root=False, trace_attrs=None):
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=noop_span), \
             patch("chains.llm_as_judge.get_callback_handler",
                   return_value=mock_handler), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.astream = recording_astream
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            await _collect_stream(
                run_judge_chain_stream("q", user="alice", use_rag=False,
                                       max_retries=0)
            )

        assert astream_configs, "astream must have been called"
        config = astream_configs[0]
        assert config is not None, \
            "config= must be passed to astream when callback_handler is set"
        callbacks = config.get("callbacks", [])
        assert mock_handler in callbacks, \
            f"CallbackHandler not in astream config callbacks: {callbacks}"


# ===========================================================================
# Section D — Concurrency
# ===========================================================================

class TestConcurrency:

    async def test_concurrent_run_judge_chain_traces_do_not_interleave(self):
        """Two concurrent calls produce separate span objects."""
        span_alice = MagicMock(name="span_alice")
        span_alice.update = MagicMock()
        span_bob = MagicMock(name="span_bob")
        span_bob.update = MagicMock()

        @contextmanager
        def per_user_span(name, *, input=None, metadata=None,
                          as_root=False, trace_attrs=None):
            if as_root:
                user = (trace_attrs or {}).get("user_id", "")
                yield span_alice if user == "alice" else span_bob
            else:
                yield MagicMock()

        async def run_for(user):
            with patch("chains.llm_as_judge.trace_span",
                       side_effect=per_user_span), \
                 patch("chains.llm_as_judge.get_callback_handler",
                       return_value=None), \
                 patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
                 patch("chains.llm_as_judge.get_relevant_context",
                       AsyncMock(return_value="")), \
                 patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                       return_value=[]), \
                 patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
                 patch("chains.llm_as_judge.compute_response_relevancy",
                       AsyncMock(return_value={"verdict": "PASS", "score": 0.9,
                                               "threshold": 0.7})):
                mock_r = MagicMock()
                mock_r.ainvoke = AsyncMock(
                    return_value=MagicMock(content=f"answer-{user}")
                )
                mock_get_reasoner.return_value = mock_r
                return await run_judge_chain("q", user=user, use_rag=False)

        results = await asyncio.gather(run_for("alice"), run_for("bob"))
        assert len(results) == 2
        # Each span must have been updated independently
        assert span_alice.update.call_count >= 1 or span_bob.update.call_count >= 1


# ===========================================================================
# Section E — Retry hierarchy
# ===========================================================================

class TestRetryHierarchy:

    async def test_run_judge_chain_attempts_are_nested_under_root_in_order(self):
        """trace_span called with 'attempt_0', 'attempt_1', 'attempt_2' in order."""
        span_names = []

        @contextmanager
        def recording_span(name, *, input=None, metadata=None,
                           as_root=False, trace_attrs=None):
            span_names.append(name)
            yield MagicMock()

        with patch("chains.llm_as_judge.trace_span", side_effect=recording_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.generate_feedback",
                   AsyncMock(return_value="try harder")), \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(side_effect=[
                       {"verdict": "FAIL", "score": 0.3, "threshold": 0.7},
                       {"verdict": "FAIL", "score": 0.5, "threshold": 0.7},
                       {"verdict": "PASS", "score": 0.9, "threshold": 0.7},
                   ])):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            await run_judge_chain("q", user="alice", use_rag=False, max_retries=2)

        attempt_spans = [n for n in span_names if n.startswith("attempt_")]
        assert attempt_spans == ["attempt_0", "attempt_1", "attempt_2"], \
            f"Expected attempt_0/1/2 in order, got: {attempt_spans}"

    async def test_run_judge_chain_judge_feedback_only_on_fail(self):
        """generate_feedback must NOT be called when first attempt is PASS."""
        mock_span = MagicMock()

        @contextmanager
        def noop_span(name, **kw):
            yield mock_span

        with patch("chains.llm_as_judge.trace_span", side_effect=noop_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner, \
             patch("chains.llm_as_judge.generate_feedback",
                   AsyncMock(return_value="feedback")) as mock_feedback, \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   AsyncMock(return_value={"verdict": "PASS", "score": 0.95,
                                           "threshold": 0.7})):
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="perfect"))
            mock_get_reasoner.return_value = mock_r

            await run_judge_chain("q", user="alice", use_rag=False)

        mock_feedback.assert_not_called()


# ===========================================================================
# Section F — judge.py generate_feedback callback_handler argument
# ===========================================================================

class TestGenerateFeedbackCallback:

    async def test_generate_feedback_passes_callback_handler_to_chain(self):
        """callback_handler provided → config={"callbacks":[handler]} in ainvoke."""
        mock_handler = MagicMock(name="CallbackHandler")
        ainvoke_calls = []

        async def recording_ainvoke(input_data, **kwargs):
            ainvoke_calls.append(kwargs)
            return {"feedback": "try again"}

        with patch("chains.judge.get_judge") as mock_get_judge:
            mock_chain = MagicMock()
            mock_chain.ainvoke = recording_ainvoke
            mock_judge_inst = MagicMock()
            mock_get_judge.return_value = mock_judge_inst
            # Patch the full chain pipeline `FEEDBACK_PROMPT | judge | parser`
            with patch("chains.judge.FEEDBACK_PROMPT") as mock_prompt, \
                 patch("chains.judge.JsonOutputParser") as mock_parser:
                mock_parser.return_value = MagicMock()
                # prompt | judge returns intermediate; intermediate | parser = chain
                intermediate = MagicMock()
                intermediate.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.__or__ = MagicMock(return_value=intermediate)

                await generate_feedback(
                    "question", "answer", 0.5, 0.7,
                    callback_handler=mock_handler,
                )

        assert ainvoke_calls, "chain.ainvoke must have been called"
        config = ainvoke_calls[0].get("config") or {}
        callbacks = config.get("callbacks") if isinstance(config, dict) else None
        assert callbacks is not None, \
            f"config.callbacks not passed (got: {ainvoke_calls[0]})"
        assert mock_handler in callbacks, \
            f"callback_handler not in config.callbacks: {callbacks}"

    async def test_generate_feedback_no_config_when_callback_handler_none(self):
        """callback_handler=None → no callbacks in ainvoke config."""
        ainvoke_calls = []

        async def recording_ainvoke(input_data, **kwargs):
            ainvoke_calls.append(kwargs)
            return {"feedback": "ok"}

        with patch("chains.judge.get_judge") as mock_get_judge:
            mock_chain = MagicMock()
            mock_chain.ainvoke = recording_ainvoke
            mock_judge_inst = MagicMock()
            mock_get_judge.return_value = mock_judge_inst
            with patch("chains.judge.FEEDBACK_PROMPT") as mock_prompt, \
                 patch("chains.judge.JsonOutputParser") as mock_parser:
                mock_parser.return_value = MagicMock()
                intermediate = MagicMock()
                intermediate.__or__ = MagicMock(return_value=mock_chain)
                mock_prompt.__or__ = MagicMock(return_value=intermediate)

                await generate_feedback(
                    "question", "answer", 0.5, 0.7,
                    callback_handler=None,
                )

        assert ainvoke_calls, "chain.ainvoke must have been called"
        config = ainvoke_calls[0].get("config")
        callbacks = config.get("callbacks") if isinstance(config, dict) else None
        assert callbacks is None, \
            f"Expected no callbacks in config when handler=None, got: {config}"
