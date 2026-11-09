from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    text: str


def _read_pdf_bytes(data: bytes) -> str:
    import PyPDF2

    reader = PyPDF2.PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return "\n".join(parts)


def _read_docx_bytes(data: bytes) -> str:
    from docx import Document

    doc = Document(BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(parts)


def parse_upload(filename: str, data: bytes) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf_bytes(data)
    elif suffix == ".docx":
        text = _read_docx_bytes(data)
    elif suffix in {".txt", ""}:
        # Best-effort decoding; resume text can be messy.
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported resume format: {suffix or '(no extension)'}")

    return ParsedDocument(filename=filename, text=text)

