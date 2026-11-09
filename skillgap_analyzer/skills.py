from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .text import normalize_text


@dataclass(frozen=True)
class SkillMatch:
    canonical: str
    found_as: str


def default_skill_ontology() -> dict[str, set[str]]:
    """Canonical skill -> set of surface forms (synonyms)."""
    return {
        "python": {"python", "python3", "python 3"},
        "java": {"java"},
        "c++": {"c++", "cplusplus", "c plus plus"},
        "c#": {"c#", "c sharp", "csharp"},
        "javascript": {"javascript", "js"},
        "typescript": {"typescript", "ts"},
        "html": {"html", "html5"},
        "css": {"css", "css3"},
        "react": {"react", "reactjs", "react.js"},
        "node.js": {"node", "nodejs", "node.js"},
        "express": {"express", "expressjs"},
        "sql": {"sql", "mysql", "postgres", "postgresql", "sqlite", "mssql"},
        "mongodb": {"mongodb", "mongo"},
        "git": {"git", "github", "gitlab"},
        "linux": {"linux", "ubuntu"},
        "docker": {"docker"},
        "kubernetes": {"kubernetes", "k8s"},
        "aws": {"aws", "amazon web services"},
        "azure": {"azure", "microsoft azure"},
        "gcp": {"gcp", "google cloud", "google cloud platform"},
        "pandas": {"pandas"},
        "numpy": {"numpy"},
        "scikit-learn": {"scikit learn", "sklearn", "scikit-learn"},
        "machine learning": {"machine learning", "ml"},
        "deep learning": {"deep learning", "dl"},
        "nlp": {"nlp", "natural language processing"},
        "computer vision": {"computer vision", "cv"},
        "statistics": {"statistics", "statistical analysis"},
        "data visualization": {"data visualization", "dataviz", "data viz", "visualization"},
        "matplotlib": {"matplotlib"},
        "plotly": {"plotly"},
        "power bi": {"power bi", "powerbi"},
        "tableau": {"tableau"},
        "excel": {"excel", "ms excel", "microsoft excel"},
        "communication": {"communication", "presentation", "stakeholder management"},
        "problem solving": {"problem solving", "problem-solving"},
        "teamwork": {"teamwork", "collaboration"},
        "figma": {"figma"},
        "rest api": {"rest", "rest api", "restful"},
        "fastapi": {"fastapi"},
        "flask": {"flask"},
        "django": {"django"},
        "streamlit": {"streamlit"},
        "tensorflow": {"tensorflow", "tf"},
        "pytorch": {"pytorch"},
    }


def _compile_patterns(ontology: dict[str, set[str]]) -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for canonical, forms in ontology.items():
        patterns: list[re.Pattern[str]] = []
        for form in forms:
            f = normalize_text(form)
            if f in {"c++", "c#"}:
                # Word boundaries do not work well with +/#; use conservative regex.
                patterns.append(re.compile(rf"(?:^|\s){re.escape(f)}(?:\s|$)"))
            else:
                patterns.append(re.compile(rf"\b{re.escape(f)}\b"))
        compiled[canonical] = patterns
    return compiled


def extract_skills(text: str, *, ontology: dict[str, set[str]] | None = None) -> tuple[set[str], list[SkillMatch]]:
    ontology = ontology or default_skill_ontology()
    compiled = _compile_patterns(ontology)
    normalized = normalize_text(text)

    found: set[str] = set()
    matches: list[SkillMatch] = []

    for canonical, patterns in compiled.items():
        for pattern in patterns:
            m = pattern.search(normalized)
            if m:
                found.add(canonical)
                matches.append(SkillMatch(canonical=canonical, found_as=m.group(0).strip()))
                break

    # Add some light heuristics for common phrases.
    if "data analysis" in normalized:
        found.add("statistics")
    return found, matches


def skills_to_text(skills: Iterable[str]) -> str:
    return " ".join(sorted(set(skills)))
