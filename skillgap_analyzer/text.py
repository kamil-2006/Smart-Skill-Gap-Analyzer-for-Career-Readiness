from __future__ import annotations

import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")

# Minimal stopword list to keep the project fully offline (no NLTK downloads).
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.lower()
    # Keep characters common in skills (c++, c#, node.js, etc.)
    text = re.sub(r"[^a-z0-9\+\#\.\s\-]", " ", text)
    text = text.replace("-", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-z0-9\+\#\.]+", text)
    return [t for t in tokens if t and t not in _STOPWORDS]


def as_space_joined(tokens: Iterable[str]) -> str:
    return " ".join(tokens)

