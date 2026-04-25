"""
Unit tests for Phase 4F: extract_headers (app/rag/headers.py).

Tests cover:
  - Normal: PDF/MD/DOCX/PPTX/XLSX/TXT/DOC header extraction
  - Error: unknown loader_type, empty documents list
  - Edge: whitespace-only title, multiple Title elements
  - Idempotency: repeated calls return identical results

Source (app/rag/headers.py) does not exist yet — ImportError -> pytest.skip.
Tests will activate automatically once the module is implemented.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    from rag.headers import extract_headers
except ImportError:
    extract_headers = None


def _skip_if_missing():
    if extract_headers is None:
        pytest.skip("rag.headers not implemented yet")


# ---------------------------------------------------------------------------
# Helper: build a mock LangChain Document
# ---------------------------------------------------------------------------


def _doc(content="content", **meta):
    """Create a MagicMock Document with page_content and metadata."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = meta
    return doc


# ===========================================================================
# Normal cases
# ===========================================================================


class TestNormalCases:
    def test_pdf_title_from_metadata(self, tmp_path):
        """PDF: metadata['title'] non-empty -> returned as title."""
        _skip_if_missing()
        pdf = tmp_path / "report.pdf"
        pdf.touch()
        docs = [_doc(content="page1", title="Annual Report", category="NarrativeText")]
        title, sections = extract_headers(pdf, docs, "pdf")
        assert title == "Annual Report"

    def test_pdf_title_empty_falls_back_to_stem(self, tmp_path):
        """PDF: empty title in metadata -> file_path.stem as title."""
        _skip_if_missing()
        pdf = tmp_path / "my_document.pdf"
        pdf.touch()
        docs = [_doc(content="page1", title="", category="NarrativeText")]
        title, sections = extract_headers(pdf, docs, "pdf")
        assert title == "my_document"

    def test_md_title_from_category_title_element(self, tmp_path):
        """MD: first element with category='Title' -> its text is the title."""
        _skip_if_missing()
        md = tmp_path / "guide.md"
        md.touch()
        docs = [
            _doc(content="Introduction", category="Title"),
            _doc(content="Body text", category="NarrativeText"),
        ]
        title, sections = extract_headers(md, docs, "md")
        assert title == "Introduction"

    def test_md_no_title_element_falls_back_to_stem(self, tmp_path):
        """MD: no Title element -> file_path.stem."""
        _skip_if_missing()
        md = tmp_path / "readme_notes.md"
        md.touch()
        docs = [
            _doc(content="Just text", category="NarrativeText"),
        ]
        title, sections = extract_headers(md, docs, "md")
        assert title == "readme_notes"

    def test_md_category_depth_reflected_in_section_path(self, tmp_path):
        """MD: element with category_depth -> section_path is non-None for that doc."""
        _skip_if_missing()
        md = tmp_path / "doc.md"
        md.touch()
        docs = [
            _doc(content="Title Text", category="Title"),
            _doc(content="Section 1", category="Title", category_depth=1),
            _doc(content="Body", category="NarrativeText"),
        ]
        title, sections = extract_headers(md, docs, "md")
        assert len(sections) == len(docs)
        # The element with category_depth should have a non-None section_path
        assert sections[1] is not None
        assert "Section 1" in sections[1]

    def test_docx_title_style_element(self, tmp_path):
        """DOCX: element with category='Title' -> title is that text."""
        _skip_if_missing()
        docx = tmp_path / "proposal.docx"
        docx.touch()
        docs = [
            _doc(content="Project Proposal", category="Title"),
            _doc(content="Executive Summary", category="NarrativeText"),
        ]
        title, sections = extract_headers(docx, docs, "docx")
        assert title == "Project Proposal"

    def test_pptx_per_slide_section_path(self, tmp_path):
        """PPTX: each slide produces a section_path containing slide number info."""
        _skip_if_missing()
        pptx = tmp_path / "slides.pptx"
        pptx.touch()
        docs = [
            _doc(content="Slide 1 content", category="Title", page_name="Slide 1"),
            _doc(content="Slide 2 content", category="NarrativeText", page_name="Slide 2"),
        ]
        title, sections = extract_headers(pptx, docs, "pptx")
        assert len(sections) == 2
        # Each slide should have a section_path that is not None
        for i, sec in enumerate(sections):
            assert sec is not None, f"Slide {i+1} section_path should not be None"

    def test_xlsx_page_name_in_section_path(self, tmp_path):
        """XLSX: page_name metadata -> reflected in section_path."""
        _skip_if_missing()
        xlsx = tmp_path / "data.xlsx"
        xlsx.touch()
        docs = [
            _doc(content="Row data", category="NarrativeText", page_name="Sheet1"),
            _doc(content="More data", category="NarrativeText", page_name="Sheet2"),
        ]
        title, sections = extract_headers(xlsx, docs, "xlsx")
        assert len(sections) == 2
        assert sections[0] is not None
        assert "Sheet1" in sections[0]
        assert sections[1] is not None
        assert "Sheet2" in sections[1]

    def test_txt_title_is_stem_and_all_sections_none(self, tmp_path):
        """TXT: title=stem, all section_paths=None."""
        _skip_if_missing()
        txt = tmp_path / "notes.txt"
        txt.touch()
        docs = [
            _doc(content="Line 1"),
            _doc(content="Line 2"),
        ]
        title, sections = extract_headers(txt, docs, "txt")
        assert title == "notes"
        assert len(sections) == len(docs)
        assert all(s is None for s in sections)

    def test_doc_title_is_stem_and_all_sections_none(self, tmp_path):
        """DOC (legacy Word): title=stem, all section_paths=None."""
        _skip_if_missing()
        doc_file = tmp_path / "legacy_report.doc"
        doc_file.touch()
        docs = [
            _doc(content="Content A"),
            _doc(content="Content B"),
        ]
        title, sections = extract_headers(doc_file, docs, "doc")
        assert title == "legacy_report"
        assert len(sections) == len(docs)
        assert all(s is None for s in sections)


# ===========================================================================
# Error cases
# ===========================================================================


class TestErrorCases:
    def test_unknown_loader_type_raises_value_error(self, tmp_path):
        """Unknown loader_type -> ValueError."""
        _skip_if_missing()
        f = tmp_path / "file.xyz"
        f.touch()
        docs = [_doc(content="some content")]
        with pytest.raises(ValueError):
            extract_headers(f, docs, "xyz_unknown")

    def test_empty_documents_returns_stem_and_empty_list(self, tmp_path):
        """documents=[] -> (file_path.stem, [])."""
        _skip_if_missing()
        pdf = tmp_path / "empty_file.pdf"
        pdf.touch()
        title, sections = extract_headers(pdf, [], "pdf")
        assert title == "empty_file"
        assert sections == []


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_pdf_whitespace_only_title_falls_back_to_stem(self, tmp_path):
        """PDF: title='   ' (whitespace only) -> file_path.stem fallback."""
        _skip_if_missing()
        pdf = tmp_path / "whitespace_title.pdf"
        pdf.touch()
        docs = [_doc(content="content", title="   ", category="NarrativeText")]
        title, sections = extract_headers(pdf, docs, "pdf")
        assert title == "whitespace_title"

    def test_md_multiple_title_elements_uses_first(self, tmp_path):
        """MD: multiple Title elements -> only the first one is used as title."""
        _skip_if_missing()
        md = tmp_path / "multi_title.md"
        md.touch()
        docs = [
            _doc(content="First Title", category="Title"),
            _doc(content="Second Title", category="Title"),
            _doc(content="Body", category="NarrativeText"),
        ]
        title, sections = extract_headers(md, docs, "md")
        assert title == "First Title"
        assert "Second Title" not in title


# ===========================================================================
# Idempotency
# ===========================================================================


class TestIdempotency:
    def test_same_input_twice_returns_identical_result(self, tmp_path):
        """Same arguments twice -> identical (title, sections) pairs."""
        _skip_if_missing()
        md = tmp_path / "idem.md"
        md.touch()
        docs = [
            _doc(content="My Title", category="Title"),
            _doc(content="Section 1", category="Title", category_depth=1),
            _doc(content="Body text", category="NarrativeText"),
        ]

        result1 = extract_headers(md, docs, "md")
        # Deep-copy to ensure we're not relying on mutable state
        result2 = extract_headers(copy.deepcopy(md), copy.deepcopy(docs), "md")

        assert result1[0] == result2[0], "title must be deterministic"
        assert result1[1] == result2[1], "section_paths must be deterministic"

    def test_pdf_idempotency(self, tmp_path):
        """PDF extract_headers produces same output on repeated calls."""
        _skip_if_missing()
        pdf = tmp_path / "stable.pdf"
        pdf.touch()
        docs = [_doc(content="page", title="Stable Title", category="NarrativeText")]

        r1 = extract_headers(pdf, docs, "pdf")
        r2 = extract_headers(pdf, docs, "pdf")
        assert r1 == r2
