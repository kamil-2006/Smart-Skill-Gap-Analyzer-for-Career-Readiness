from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class GapResult:
    matched: set[str]
    missing: set[str]
    extra: set[str]
    coverage: float
    cosine: float


def _binary_vectors(user: set[str], target: set[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vocab = sorted(user | target)
    u = np.array([1.0 if s in user else 0.0 for s in vocab], dtype=float).reshape(1, -1)
    t = np.array([1.0 if s in target else 0.0 for s in vocab], dtype=float).reshape(1, -1)
    return u, t, vocab


def analyze_gap(user_skills: set[str], target_skills: set[str]) -> GapResult:
    matched = set(user_skills & target_skills)
    missing = set(target_skills - user_skills)
    extra = set(user_skills - target_skills)
    coverage = (len(matched) / len(target_skills)) if target_skills else 0.0

    if not (user_skills or target_skills):
        cosine = 0.0
    else:
        u, t, _ = _binary_vectors(user_skills, target_skills)
        cosine = float(cosine_similarity(u, t)[0, 0])

    return GapResult(
        matched=matched,
        missing=missing,
        extra=extra,
        coverage=float(coverage),
        cosine=cosine,
    )

