"""
Unit tests for Phase 5 Step 2c: ContextPrecision batch evaluation via Langfuse Datasets.

Tests app/evaluation/run_cp_eval.py — the offline batch evaluation script.
All Langfuse, LLM, and RAG calls are mocked; no server required.

Test groups:
  A. _item_id — deterministic ID generation
  B. _load_queries — YAML loading
  C. _upsert_dataset_item — Langfuse dataset item creation
  D. _generate_answer — LLM answer generation
  E. _evaluate_item — CP scoring + Evaluation object construction
  F. run_eval — full evaluation loop integration (sync, run_experiment)
  G. Edge cases — empty queries, score=0.0, task/evaluator callables
"""
import hashlib
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        "langfuse", "langfuse.experiment",
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
# E. _evaluate_item — CP scoring + Evaluation object construction
# ===========================================================================

class TestEvaluateItem:
    def setup_method(self):
        self.mod = _import_module()
        # Reset Evaluation mock call count between tests
        if "langfuse.experiment" in sys.modules:
            sys.modules["langfuse.experiment"].Evaluation.reset_mock()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_score(self):
        """Returns [] when CP produces no score (no reference)."""
        with patch("evaluation.run_cp_eval.compute_context_precision",
                   new_callable=AsyncMock, return_value={"score": None}):
            result = await self.mod._evaluate_item("q", "ctx", "ans", "")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_single_evaluation_when_score_exists(self):
        """Returns a list with one Evaluation when CP produces a score."""
        with patch("evaluation.run_cp_eval.compute_context_precision",
                   new_callable=AsyncMock, return_value={"score": 0.85}):
            result = await self.mod._evaluate_item("q", "ctx", "ans", "ref")
        assert len(result) == 1
        Evaluation_mock = sys.modules["langfuse.experiment"].Evaluation
        Evaluation_mock.assert_called_with(name="context_precision", value=0.85)

    @pytest.mark.asyncio
    async def test_score_value_in_evaluation(self):
        """Various score values are passed verbatim to Evaluation."""
        for score_val in [0.0, 0.5, 1.0]:
            sys.modules["langfuse.experiment"].Evaluation.reset_mock()
            with patch("evaluation.run_cp_eval.compute_context_precision",
                       new_callable=AsyncMock, return_value={"score": score_val}):
                result = await self.mod._evaluate_item("q", "ctx", "ans", "ref")
            assert len(result) == 1
            sys.modules["langfuse.experiment"].Evaluation.assert_called_with(
                name="context_precision", value=score_val
            )


# ===========================================================================
# F. run_eval — full evaluation loop integration (sync, run_experiment)
# ===========================================================================

class TestRunEval:
    def setup_method(self):
        self.mod = _import_module()

    def test_run_eval_creates_dataset(self, tmp_path):
        """run_eval creates the Langfuse dataset before processing items."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text("queries: []\n", encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "my-eval-dataset"
        args.run_name = "run-v1"
        args.user = "nire"

        mock_langfuse = MagicMock()
        mock_langfuse.get_dataset.return_value = MagicMock(items=[])

        with patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse):
            self.mod.run_eval(args)

        mock_langfuse.create_dataset.assert_called_once_with(name="my-eval-dataset")

    def test_run_eval_calls_run_experiment_with_correct_args(self, tmp_path):
        """run_eval calls run_experiment with name, run_name, data, task, evaluators."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            queries:
              - query: "Q1"
                reference: "R1"
        """), encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "test-ds"
        args.run_name = "run-001"
        args.user = "nire"

        mock_langfuse = MagicMock()
        sentinel_items = [MagicMock()]
        mock_langfuse.get_dataset.return_value = MagicMock(items=sentinel_items)

        with patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse):
            self.mod.run_eval(args)

        mock_langfuse.run_experiment.assert_called_once()
        kwargs = mock_langfuse.run_experiment.call_args.kwargs
        assert kwargs["name"] == "test-ds"
        assert kwargs["run_name"] == "run-001"
        assert kwargs["data"] is sentinel_items

    def test_run_eval_flushes_langfuse(self, tmp_path):
        """run_eval always calls langfuse.flush() even with no items."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text("queries: []\n", encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "test-ds"
        args.run_name = "run-001"
        args.user = "nire"

        mock_langfuse = MagicMock()
        mock_langfuse.get_dataset.return_value = MagicMock(items=[])

        with patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse):
            self.mod.run_eval(args)

        mock_langfuse.flush.assert_called_once()

    def test_run_eval_flushes_on_exception(self, tmp_path):
        """langfuse.flush() is called even if an error occurs during item upsert."""
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

        with patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse):
            with pytest.raises(RuntimeError):
                self.mod.run_eval(args)

        mock_langfuse.flush.assert_called_once()


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
    async def test_evaluate_item_score_zero_not_skipped(self):
        """Score=0.0 is a valid score and must be included, not treated as falsy."""
        if "langfuse.experiment" in sys.modules:
            sys.modules["langfuse.experiment"].Evaluation.reset_mock()
        with patch("evaluation.run_cp_eval.compute_context_precision",
                   new_callable=AsyncMock, return_value={"score": 0.0}):
            result = await self.mod._evaluate_item("q", "ctx", "ans", "ref")
        assert len(result) == 1
        sys.modules["langfuse.experiment"].Evaluation.assert_called_with(
            name="context_precision", value=0.0
        )

    def test_run_eval_task_and_evaluators_are_callable(self, tmp_path):
        """run_eval passes callable task and non-empty evaluators list to run_experiment."""
        yaml_file = tmp_path / "q.yaml"
        yaml_file.write_text("queries: []\n", encoding="utf-8")

        args = MagicMock()
        args.queries = str(yaml_file)
        args.dataset = "ds"
        args.run_name = "run"
        args.user = "nire"

        mock_langfuse = MagicMock()
        mock_langfuse.get_dataset.return_value = MagicMock(items=[])

        with patch("evaluation.run_cp_eval._get_langfuse", return_value=mock_langfuse):
            self.mod.run_eval(args)

        kwargs = mock_langfuse.run_experiment.call_args.kwargs
        assert callable(kwargs["task"]), "task must be callable"
        assert len(kwargs["evaluators"]) == 1
        assert callable(kwargs["evaluators"][0]), "evaluators[0] must be callable"
