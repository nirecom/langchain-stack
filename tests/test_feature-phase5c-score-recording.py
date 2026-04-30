"""
Unit tests for Phase 5 Step 2c: ContextPrecision batch evaluation via Langfuse Datasets.

Tests app/evaluation/run_cp_eval.py — the offline batch evaluation script.
All Langfuse, LLM, and RAG calls are mocked; no server required.

Test groups:
  A. _item_id — deterministic ID generation
  B. _load_queries — YAML loading
  C. _upsert_dataset_item — Langfuse dataset item creation
  D. _generate_answer — LLM answer generation
  E. _evaluate_item — context retrieval + CP scoring + span recording
  F. run_eval — full evaluation loop integration
  G. Edge cases — empty queries, no reference, SKIP result
"""
import hashlib
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

_APP_DIR = str(Path(__file__).resolve().parents[1] / "app")


def _import_module():
    """Import run_cp_eval with heavy dependencies mocked."""
    import importlib

    if _APP_DIR not in sys.path:
        sys.path.insert(0, _APP_DIR)

    # Mock external dependencies (not the evaluation package itself —
    # mocking a parent package as MagicMock breaks submodule imports)
    for mod in (
        "langfuse",
        "rag", "rag.retriever",
        "models", "models.provider",
        "langchain_core", "langchain_core.messages",
        "settings",
    ):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # Mock evaluation.metrics separately so it doesn't try to load ragas
    if "evaluation.metrics" not in sys.modules:
        sys.modules["evaluation.metrics"] = MagicMock()

    # Force fresh import of the module under test
    for key in list(sys.modules):
        if "run_cp_eval" in key:
            del sys.modules[key]

    return importlib.import_module("evaluation.run_cp_eval")


# ===========================================================================
# A. _item_id — deterministic ID generation
# ===========================================================================

class TestItemId:
    def setup_method(self):
        self.mod = _import_module()

    def test_deterministic(self):
        """Same query+reference always produces the same ID."""
        id1 = self.mod._item_id("What is X?", "X is Y.")
        id2 = self.mod._item_id("What is X?", "X is Y.")
        assert id1 == id2

    def test_length_16_hex(self):
        """ID is exactly 16 hexadecimal characters."""
        item_id = self.mod._item_id("query", "reference")
        assert len(item_id) == 16
        assert all(c in "0123456789abcdef" for c in item_id)

    def test_different_query_produces_different_id(self):
        """Different queries produce different IDs."""
        assert self.mod._item_id("A", "ref") != self.mod._item_id("B", "ref")

    def test_different_reference_produces_different_id(self):
        """Different references produce different IDs."""
        assert self.mod._item_id("q", "ref1") != self.mod._item_id("q", "ref2")

    def test_empty_inputs_stable(self):
        """Empty query and reference produce a deterministic 16-char ID."""
        item_id = self.mod._item_id("", "")
        assert len(item_id) == 16
        assert self.mod._item_id("", "") == item_id

    def test_matches_sha256_prefix(self):
        """ID matches the first 16 chars of sha256(query+reference)."""
        query, reference = "test query", "test reference"
        expected = hashlib.sha256(f"{query}{reference}".encode()).hexdigest()[:16]
        assert self.mod._item_id(query, reference) == expected


# ===========================================================================
# B. _load_queries — YAML loading
# ===========================================================================

class TestLoadQueries:
    def setup_method(self):
        self.mod = _import_module()

    def test_load_queries_basic(self, tmp_path):
        """Loads a YAML file with queries list."""
        yaml_file = tmp_path / "queries.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            queries:
              - query: "What is X?"
                reference: "X is Y."
              - query: "How does Z work?"
                reference: "Z works by..."
        """), encoding="utf-8")
        result = self.mod._load_queries(str(yaml_file))
        assert len(result) == 2
        assert result[0]["query"] == "What is X?"
        assert result[0]["reference"] == "X is Y."

    def test_load_queries_empty(self, tmp_path):
        """YAML with no queries key returns empty list."""
        yaml_file = tmp_path / "queries.yaml"
        yaml_file.write_text("other_key: value\n", encoding="utf-8")
        result = self.mod._load_queries(str(yaml_file))
        assert result == []

    def test_load_queries_empty_list(self, tmp_path):
        """YAML with empty queries list returns empty list."""
        yaml_file = tmp_path / "queries.yaml"
        yaml_file.write_text("queries: []\n", encoding="utf-8")
        result = self.mod._load_queries(str(yaml_file))
        assert result == []

    def test_load_queries_no_reference(self, tmp_path):
        """Queries without reference field are included as-is."""
        yaml_file = tmp_path / "queries.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            queries:
              - query: "What is X?"
        """), encoding="utf-8")
        result = self.mod._load_queries(str(yaml_file))
        assert len(result) == 1
        assert result[0]["query"] == "What is X?"
        assert "reference" not in result[0]


# ===========================================================================
# C. _upsert_dataset_item — Langfuse dataset item creation
# ===========================================================================

class TestUpsertDatasetItem:
    def setup_method(self):
        self.mod = _import_module()

    def test_calls_create_dataset_item(self):
        """Calls langfuse.create_dataset_item with correct fields."""
        mock_langfuse = MagicMock()
        item_data = {"query": "What is X?", "reference": "X is Y."}
        self.mod._upsert_dataset_item(mock_langfuse, "my-dataset", item_data)
        mock_langfuse.create_dataset_item.assert_called_once()
        call_kwargs = mock_langfuse.create_dataset_item.call_args.kwargs
        assert call_kwargs["dataset_name"] == "my-dataset"
        assert call_kwargs["input"] == "What is X?"
        assert call_kwargs["expected_output"] == "X is Y."

    def test_deterministic_id_on_upsert(self):
        """The ID passed to create_dataset_item is the same as _item_id()."""
        mock_langfuse = MagicMock()
        item_data = {"query": "q", "reference": "r"}
        self.mod._upsert_dataset_item(mock_langfuse, "ds", item_data)
        call_kwargs = mock_langfuse.create_dataset_item.call_args.kwargs
        expected_id = self.mod._item_id("q", "r")
        assert call_kwargs["id"] == expected_id

    def test_empty_reference_defaults_to_empty_string(self):
        """Item without reference uses empty string for expected_output."""
        mock_langfuse = MagicMock()
        item_data = {"query": "q"}
        self.mod._upsert_dataset_item(mock_langfuse, "ds", item_data)
        call_kwargs = mock_langfuse.create_dataset_item.call_args.kwargs
        assert call_kwargs["expected_output"] == ""

    def test_idempotent_same_item_twice(self):
        """Calling upsert twice for the same item passes the same ID."""
        mock_langfuse = MagicMock()
        item_data = {"query": "q", "reference": "r"}
        self.mod._upsert_dataset_item(mock_langfuse, "ds", item_data)
        self.mod._upsert_dataset_item(mock_langfuse, "ds", item_data)
        calls = mock_langfuse.create_dataset_item.call_args_list
        assert calls[0].kwargs["id"] == calls[1].kwargs["id"]


# ===========================================================================
# D. _generate_answer — LLM answer generation
# ===========================================================================

class TestGenerateAnswer:
    def setup_method(self):
        self.mod = _import_module()

    @pytest.mark.asyncio
    async def test_returns_llm_content(self):
        """Returns the content from the LLM response."""
        mock_msg = MagicMock()
        mock_msg.content = "The answer is 42."
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_msg

        # get_judge is a lazy import inside _generate_answer; patch via models.provider
        sys.modules["models.provider"].get_judge = MagicMock(return_value=mock_llm)
        result = await self.mod._generate_answer("What is X?", "Context: X is 42.")
        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_calls_ainvoke_with_messages(self):
        """Calls llm.ainvoke once with a two-element message list."""
        mock_msg = MagicMock()
        mock_msg.content = "answer"
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_msg

        sys.modules["models.provider"].get_judge = MagicMock(return_value=mock_llm)
        await self.mod._generate_answer("question", "context text")

        assert mock_llm.ainvoke.call_count == 1
        messages = mock_llm.ainvoke.call_args.args[0]
        # Should be a list of two messages (SystemMessage + HumanMessage)
        assert isinstance(messages, list)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_empty_context_still_calls_llm(self):
        """Empty context: LLM is still called, returns its content."""
        mock_msg = MagicMock()
        mock_msg.content = "I don't know."
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_msg

        sys.modules["models.provider"].get_judge = MagicMock(return_value=mock_llm)
        result = await self.mod._generate_answer("question", "")
        assert result == "I don't know."


# ===========================================================================
# E. _evaluate_item — context retrieval + CP scoring + span recording
# ===========================================================================

class TestEvaluateItem:
    def setup_method(self):
        self.mod = _import_module()

    @pytest.mark.asyncio
    async def test_calls_span_score_on_success(self):
        """When CP returns a score, span.score is called with context_precision."""
        mock_span = MagicMock()
        mock_item = MagicMock()
        mock_item.input = "What is X?"
        mock_item.expected_output = "X is Y."
        mock_item.id = "abc123"

        with (
            patch("evaluation.run_cp_eval.get_relevant_context",
                  new_callable=AsyncMock, return_value="Context about X."),
            patch("evaluation.run_cp_eval._generate_answer",
                  new_callable=AsyncMock, return_value="X is Y because..."),
            patch("evaluation.run_cp_eval.compute_context_precision",
                  new_callable=AsyncMock, return_value={"score": 0.85}),
        ):
            await self.mod._evaluate_item(mock_item, span=mock_span, user="nire")

        mock_span.score.assert_called_once_with(name="context_precision", value=0.85)

    @pytest.mark.asyncio
    async def test_skips_span_score_when_no_score(self):
        """When CP returns None score (no reference), span.score is NOT called."""
        mock_span = MagicMock()
        mock_item = MagicMock()
        mock_item.input = "What is X?"
        mock_item.expected_output = ""
        mock_item.id = "abc123"

        with (
            patch("evaluation.run_cp_eval.get_relevant_context",
                  new_callable=AsyncMock, return_value="some context"),
            patch("evaluation.run_cp_eval._generate_answer",
                  new_callable=AsyncMock, return_value="some answer"),
            patch("evaluation.run_cp_eval.compute_context_precision",
                  new_callable=AsyncMock, return_value={"score": None}),
        ):
            await self.mod._evaluate_item(mock_item, span=mock_span, user="nire")

        mock_span.score.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_user_to_get_relevant_context(self):
        """user arg is forwarded to get_relevant_context."""
        mock_span = MagicMock()
        mock_item = MagicMock()
        mock_item.input = "question"
        mock_item.expected_output = "ref"
        mock_item.id = "xyz"
        mock_get_context = AsyncMock(return_value="ctx")

        with (
            patch("evaluation.run_cp_eval.get_relevant_context", mock_get_context),
            patch("evaluation.run_cp_eval._generate_answer",
                  new_callable=AsyncMock, return_value="ans"),
            patch("evaluation.run_cp_eval.compute_context_precision",
                  new_callable=AsyncMock, return_value={"score": 0.5}),
        ):
            await self.mod._evaluate_item(mock_item, span=mock_span, user="kyoko")

        mock_get_context.assert_called_once_with("question", user="kyoko")

    @pytest.mark.asyncio
    async def test_score_value_passed_correctly(self):
        """CP score value is passed verbatim to span.score."""
        mock_span = MagicMock()
        mock_item = MagicMock()
        mock_item.input = "q"
        mock_item.expected_output = "ref"
        mock_item.id = "id1"

        for score_val in [0.0, 0.5, 1.0]:
            mock_span.reset_mock()
            with (
                patch("evaluation.run_cp_eval.get_relevant_context",
                      new_callable=AsyncMock, return_value="ctx"),
                patch("evaluation.run_cp_eval._generate_answer",
                      new_callable=AsyncMock, return_value="ans"),
                patch("evaluation.run_cp_eval.compute_context_precision",
                      new_callable=AsyncMock, return_value={"score": score_val}),
            ):
                await self.mod._evaluate_item(mock_item, span=mock_span, user="nire")

            mock_span.score.assert_called_once_with(name="context_precision", value=score_val)


# ===========================================================================
# F. run_eval — full evaluation loop integration
# ===========================================================================

class TestRunEval:
    def setup_method(self):
        self.mod = _import_module()

    @pytest.mark.asyncio
    async def test_run_eval_calls_evaluate_per_item(self, tmp_path):
        """run_eval calls _evaluate_item once per query in the file."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            queries:
              - query: "Q1"
                reference: "R1"
              - query: "Q2"
                reference: "R2"
        """), encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "test-ds"
        args.run_name = "run-001"
        args.user = "nire"

        mock_langfuse = MagicMock()
        mock_item = MagicMock()
        mock_item.input = "Q1"
        mock_item.expected_output = "R1"
        mock_item.id = "id1"
        mock_langfuse.create_dataset_item.return_value = mock_item

        mock_span = MagicMock()
        mock_item.run.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_item.run.return_value.__exit__ = MagicMock(return_value=False)

        evaluate_calls = []

        async def fake_evaluate(item, *, span, user):
            evaluate_calls.append((item, user))

        with (
            patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse),
            patch("evaluation.run_cp_eval._evaluate_item", fake_evaluate),
        ):
            await self.mod.run_eval(args)

        assert len(evaluate_calls) == 2

    @pytest.mark.asyncio
    async def test_run_eval_flushes_langfuse(self, tmp_path):
        """run_eval always calls langfuse.flush() even with no items."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text("queries: []\n", encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "test-ds"
        args.run_name = "run-001"
        args.user = "nire"

        mock_langfuse = MagicMock()

        with (
            patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse),
            patch("evaluation.run_cp_eval._evaluate_item", new_callable=AsyncMock),
        ):
            await self.mod.run_eval(args)

        mock_langfuse.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_eval_creates_dataset(self, tmp_path):
        """run_eval creates the Langfuse dataset before processing items."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text("queries: []\n", encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "my-eval-dataset"
        args.run_name = "run-v1"
        args.user = "nire"

        mock_langfuse = MagicMock()

        with (
            patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse),
            patch("evaluation.run_cp_eval._evaluate_item", new_callable=AsyncMock),
        ):
            await self.mod.run_eval(args)

        mock_langfuse.create_dataset.assert_called_once_with(name="my-eval-dataset")


# ===========================================================================
# G. Edge cases
# ===========================================================================

class TestEdgeCases:
    def setup_method(self):
        self.mod = _import_module()

    def test_item_id_long_strings_produce_16_chars(self):
        """Very long query+reference still produces 16-char ID."""
        long_query = "x" * 10000
        long_ref = "y" * 10000
        item_id = self.mod._item_id(long_query, long_ref)
        assert len(item_id) == 16

    @pytest.mark.asyncio
    async def test_evaluate_item_score_zero_still_records(self):
        """Score=0.0 is a valid score and must be recorded, not skipped."""
        mock_span = MagicMock()
        mock_item = MagicMock()
        mock_item.input = "q"
        mock_item.expected_output = "ref"
        mock_item.id = "id1"

        with (
            patch("evaluation.run_cp_eval.get_relevant_context",
                  new_callable=AsyncMock, return_value="ctx"),
            patch("evaluation.run_cp_eval._generate_answer",
                  new_callable=AsyncMock, return_value="ans"),
            patch("evaluation.run_cp_eval.compute_context_precision",
                  new_callable=AsyncMock, return_value={"score": 0.0}),
        ):
            await self.mod._evaluate_item(mock_item, span=mock_span, user="nire")

        mock_span.score.assert_called_once_with(name="context_precision", value=0.0)

    @pytest.mark.asyncio
    async def test_run_eval_flushes_on_exception(self, tmp_path):
        """langfuse.flush() is called even if evaluation raises an exception."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            queries:
              - query: "q"
                reference: "r"
        """), encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "ds"
        args.run_name = "run"
        args.user = "nire"

        mock_langfuse = MagicMock()
        mock_langfuse.create_dataset_item.side_effect = RuntimeError("Langfuse down")

        with (
            patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse),
        ):
            with pytest.raises(RuntimeError):
                await self.mod.run_eval(args)

        mock_langfuse.flush.assert_called_once()
