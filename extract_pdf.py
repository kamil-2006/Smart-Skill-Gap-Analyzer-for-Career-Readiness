from __future__ import annotations

import argparse
from pathlib import Path

import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a PDF (UTF-8 safe).")
    parser.add_argument("--pdf", required=True, help="Path to the PDF")
    parser.add_argument("--out", default=".local/extracted.txt", help="Output text file path")
    args = parser.parse_args()

    text = extract_text_from_pdf(args.pdf)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text):,} characters to {args.out}")


if __name__ == "__main__":
    main()
