from __future__ import annotations

import pandas as pd


def recommend_courses(courses_df: pd.DataFrame, missing_skills: set[str], *, top_k_per_skill: int = 3) -> pd.DataFrame:
    if courses_df.empty or not missing_skills:
        return pd.DataFrame(columns=list(courses_df.columns))

    df = courses_df.copy()
    df["skill_norm"] = df["skill"].str.lower().str.strip()
    wanted = {s.lower().strip() for s in missing_skills}
    df = df[df["skill_norm"].isin(wanted)]

    # Prefer beginner/intermediate first (typical student use-case).
    level_rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
    df["level_rank"] = df["level"].str.lower().map(level_rank).fillna(99).astype(int)

    out = (
        df.sort_values(["skill_norm", "level_rank", "duration_hours", "provider", "title"], ascending=True)
        .groupby("skill_norm", as_index=False)
        .head(top_k_per_skill)
        .drop(columns=["skill_norm", "level_rank"], errors="ignore")
    )
    return out.reset_index(drop=True)

