from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SimilarityResult:
    cosine: float


def tfidf_cosine(text_a: str, text_b: str) -> SimilarityResult:
    """Cosine similarity using TF-IDF (useful when you have full job descriptions)."""
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform([text_a, text_b])
    sim = float(cosine_similarity(X[0], X[1])[0, 0])
    return SimilarityResult(cosine=sim)

