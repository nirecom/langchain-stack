"""
Header extraction utilities for structured ingestion.

Extracts title and per-document section_path from loaded documents,
using format-specific logic per loader_type.
"""
from pathlib import Path


def extract_headers(
    file_path: Path,
    documents: list,
    loader_type: str,
    *,
    header_source: str = "loader",
) -> tuple[str, list]:
    """
    Returns (title, per_doc_section_paths).
    section_paths[i] is None when no section info is available for doc i.
    """
    stem = file_path.stem

    if loader_type == "pdf":
        return _extract_pdf(documents, stem, header_source)
    if loader_type == "md":
        return _extract_md(documents, stem)
    if loader_type == "docx":
        return _extract_docx(documents, stem)
    if loader_type == "pptx":
        return _extract_pptx(documents, stem)
    if loader_type == "xlsx":
        return _extract_xlsx(documents, stem)
    if loader_type in ("txt", "doc"):
        return stem, [None] * len(documents)

    raise ValueError(f"Unknown loader_type: {loader_type!r}")


def _extract_pdf(documents, stem: str, header_source: str) -> tuple[str, list]:
    if not documents:
        return stem, []
    title = ""
    for doc in documents:
        t = (doc.metadata or {}).get("title", "")
        if t and t.strip():
            title = t.strip()
            break
    if not title:
        title = stem
    return title, [None] * len(documents)


def _extract_md(documents, stem: str) -> tuple[str, list]:
    if not documents:
        return stem, []
    title = stem
    for doc in documents:
        if (doc.metadata or {}).get("category") == "Title":
            title = doc.page_content.strip() or stem
            break

    section_paths = []
    current_sections: dict[int, str] = {}
    for doc in documents:
        meta = doc.metadata or {}
        depth = meta.get("category_depth")
        text = doc.page_content.strip()
        if depth is not None and text:
            current_sections[depth] = text
            for d in list(current_sections):
                if d > depth:
                    del current_sections[d]
        if current_sections:
            path = " / ".join(current_sections[d] for d in sorted(current_sections))
            section_paths.append(path)
        else:
            section_paths.append(None)
    return title, section_paths


def _extract_docx(documents, stem: str) -> tuple[str, list]:
    if not documents:
        return stem, []
    title = stem
    for doc in documents:
        meta = doc.metadata or {}
        if meta.get("category") == "Title":
            title = doc.page_content.strip() or stem
            break
    return title, [None] * len(documents)


def _extract_pptx(documents, stem: str) -> tuple[str, list]:
    if not documents:
        return stem, []
    title = stem
    for doc in documents:
        meta = doc.metadata or {}
        if meta.get("category") == "Title":
            title = doc.page_content.strip() or stem
            break

    section_paths = []
    for doc in documents:
        meta = doc.metadata or {}
        page_num = meta.get("page_number") or meta.get("slide_number")
        page_name = meta.get("page_name", "")
        slide_title = doc.page_content.strip()[:40] if doc.page_content else ""

        if page_num:
            label = f"Slide {page_num}"
        elif page_name:
            label = page_name  # Unstructured may set page_name="Slide N"
        else:
            label = ""

        if label:
            section_paths.append(f"{label} / {slide_title}" if slide_title else label)
        else:
            section_paths.append(None)
    return title, section_paths


def _extract_xlsx(documents, stem: str) -> tuple[str, list]:
    if not documents:
        return stem, []
    section_paths = []
    for doc in documents:
        sheet = (doc.metadata or {}).get("page_name")
        section_paths.append(sheet if sheet else None)
    return stem, section_paths
