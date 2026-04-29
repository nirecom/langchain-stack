"""
Unit tests for Phase 5 Step 2b: RAGAS callback_handler tracing integration.

Verifies:
1. evaluation/metrics.py — compute_response_relevancy accepts callback_handler kwarg
   and forwards it as callbacks=[handler] to single_turn_ascore.
2. chains/llm_as_judge.py — get_callback_handler() is called inside the ragas_eval
   trace_span block and the result is forwarded to compute_response_relevancy.

Run with:
    uv run --project app pytest tests/test_feature-phase5b-ragas-tracing.py -v
"""

import asyncio
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# sys.path — ensure app/ is importable
# ---------------------------------------------------------------------------
from pathlib import Path

_APP = Path(__file__).parent.parent / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

# ---------------------------------------------------------------------------
# Import app modules
# ---------------------------------------------------------------------------
from evaluation.metrics import compute_response_relevancy
from chains.llm_as_judge import run_judge_chain, run_judge_chain_stream


# ---------------------------------------------------------------------------
# Helpers shared across sections
# ---------------------------------------------------------------------------

def _make_noop_span():
    mock_span = MagicMock(name="span")
    mock_span.update = MagicMock()

    @contextmanager
    def noop_span(name, *, input=None, metadata=None,
                  as_root=False, trace_attrs=None):
        yield mock_span

    return mock_span, noop_span


async def _collect_stream(agen):
    events = []
    async for ev in agen:
        events.append(ev)
    return events


# ===========================================================================
# Section A — compute_response_relevancy unit tests
# ===========================================================================

class TestComputeResponseRelevancy:

    async def test_a1_callback_handler_forwarded_to_single_turn_ascore(self):
        """callback_handler present → single_turn_ascore receives callbacks=[cb]."""
        mock_cb = MagicMock(name="CallbackHandler")
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.85)

        with patch("evaluation.metrics._get_metric", return_value=mock_metric), \
             patch("evaluation.metrics.settings") as mock_settings:
            mock_settings.ragas_response_relevancy_threshold = 0.7

            await compute_response_relevancy(
                "What is AI?", "AI is ...", callback_handler=mock_cb
            )

        mock_metric.single_turn_ascore.assert_called_once()
        _, kwargs = mock_metric.single_turn_ascore.call_args
        callbacks = kwargs.get("callbacks", [])
        assert mock_cb in callbacks, \
            f"callback_handler must be in callbacks kwarg, got: {callbacks}"

    async def test_a2_no_callback_handler_passes_empty_callbacks(self):
        """callback_handler omitted → single_turn_ascore receives callbacks=[]."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.80)

        with patch("evaluation.metrics._get_metric", return_value=mock_metric), \
             patch("evaluation.metrics.settings") as mock_settings:
            mock_settings.ragas_response_relevancy_threshold = 0.7

            await compute_response_relevancy("Q?", "A.")

        mock_metric.single_turn_ascore.assert_called_once()
        _, kwargs = mock_metric.single_turn_ascore.call_args
        callbacks = kwargs.get("callbacks", [])
        assert callbacks == [], \
            f"Expected empty callbacks when handler=None, got: {callbacks}"

    async def test_a3_score_above_threshold_returns_pass(self):
        """score >= threshold → verdict='PASS'."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.75)

        with patch("evaluation.metrics._get_metric", return_value=mock_metric), \
             patch("evaluation.metrics.settings") as mock_settings:
            mock_settings.ragas_response_relevancy_threshold = 0.7

            result = await compute_response_relevancy("Q?", "A.", callback_handler=None)

        assert result["verdict"] == "PASS"
        assert result["score"] == round(0.75, 4)
        assert result["threshold"] == 0.7

    async def test_a4_score_below_threshold_returns_fail(self):
        """score < threshold → verdict='FAIL'."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(return_value=0.50)

        with patch("evaluation.metrics._get_metric", return_value=mock_metric), \
             patch("evaluation.metrics.settings") as mock_settings:
            mock_settings.ragas_response_relevancy_threshold = 0.7

            result = await compute_response_relevancy("Q?", "A.", callback_handler=None)

        assert result["verdict"] == "FAIL"
        assert result["score"] == round(0.50, 4)
        assert result["threshold"] == 0.7

    async def test_a5_exception_in_single_turn_ascore_returns_fallback(self):
        """single_turn_ascore raises → fallback with score=0.0, verdict='PASS'."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(
            side_effect=RuntimeError("RAGAS internal error")
        )

        with patch("evaluation.metrics._get_metric", return_value=mock_metric), \
             patch("evaluation.metrics.settings") as mock_settings:
            mock_settings.ragas_response_relevancy_threshold = 0.7

            result = await compute_response_relevancy(
                "Q?", "A.", callback_handler=MagicMock()
            )

        assert result["score"] == 0.0
        assert result["verdict"] == "PASS"
        assert result["threshold"] == 0.7


# ===========================================================================
# Section B — llm_as_judge.py call-site tests
# ===========================================================================

class TestLlmAsJudgeCallSite:

    async def test_b1_run_judge_chain_passes_ragas_cb_to_compute(self):
        """run_judge_chain: get_callback_handler() result forwarded as callback_handler=."""
        mock_ragas_cb = MagicMock(name="RagasCallbackHandler")
        compute_calls = []

        async def recording_compute(*, question, answer, callback_handler=None):
            compute_calls.append(callback_handler)
            return {"verdict": "PASS", "score": 0.9, "threshold": 0.7}

        _, noop_span = _make_noop_span()

        with patch("chains.llm_as_judge.trace_span", side_effect=noop_span), \
             patch("chains.llm_as_judge.get_callback_handler",
                   return_value=mock_ragas_cb), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   side_effect=recording_compute), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner:
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            await run_judge_chain("What is AI?", user="alice", use_rag=False)

        assert compute_calls, "compute_response_relevancy must have been called"
        assert compute_calls[0] is mock_ragas_cb, (
            f"Expected callback_handler={mock_ragas_cb!r}, "
            f"got: {compute_calls[0]!r}"
        )

    async def test_b2_run_judge_chain_stream_passes_ragas_cb_to_compute(self):
        """run_judge_chain_stream: get_callback_handler() result forwarded as callback_handler=."""
        mock_ragas_cb = MagicMock(name="RagasCallbackHandlerStream")
        compute_calls = []

        async def recording_compute(*, question, answer, callback_handler=None):
            compute_calls.append(callback_handler)
            return {"verdict": "PASS", "score": 0.9, "threshold": 0.7}

        _, noop_span = _make_noop_span()

        async def fake_astream(messages, **kwargs):
            yield MagicMock(content="tok")

        with patch("chains.llm_as_judge.trace_span", side_effect=noop_span), \
             patch("chains.llm_as_judge.get_callback_handler",
                   return_value=mock_ragas_cb), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   side_effect=recording_compute), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner:
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_r.astream = fake_astream
            mock_get_reasoner.return_value = mock_r

            await _collect_stream(
                run_judge_chain_stream("What is AI?", user="alice",
                                       use_rag=False, max_retries=0)
            )

        assert compute_calls, "compute_response_relevancy must have been called"
        assert compute_calls[0] is mock_ragas_cb, (
            f"Expected callback_handler={mock_ragas_cb!r}, "
            f"got: {compute_calls[0]!r}"
        )

    async def test_b3_run_judge_chain_passes_none_when_tracing_disabled(self):
        """Tracing disabled (get_callback_handler returns None) → callback_handler=None."""
        compute_calls = []

        async def recording_compute(*, question, answer, callback_handler=None):
            compute_calls.append(callback_handler)
            return {"verdict": "PASS", "score": 0.9, "threshold": 0.7}

        _, noop_span = _make_noop_span()

        with patch("chains.llm_as_judge.trace_span", side_effect=noop_span), \
             patch("chains.llm_as_judge.get_callback_handler", return_value=None), \
             patch("chains.llm_as_judge.probe_endpoints", AsyncMock()), \
             patch("chains.llm_as_judge.get_relevant_context",
                   AsyncMock(return_value="")), \
             patch("chains.llm_as_judge.get_permitted_datasources_for_user",
                   return_value=[]), \
             patch("chains.llm_as_judge.compute_response_relevancy",
                   side_effect=recording_compute), \
             patch("chains.llm_as_judge.get_reasoner") as mock_get_reasoner:
            mock_r = MagicMock()
            mock_r.ainvoke = AsyncMock(return_value=MagicMock(content="answer"))
            mock_get_reasoner.return_value = mock_r

            await run_judge_chain("What is AI?", user="alice", use_rag=False)

        assert compute_calls, "compute_response_relevancy must have been called"
        assert compute_calls[0] is None, (
            f"Expected callback_handler=None when tracing disabled, "
            f"got: {compute_calls[0]!r}"
        )
