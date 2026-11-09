from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


@dataclass(frozen=True)
class JobRole:
    role: str
    skills: list[str]
    description: str | None = None
    track: str | None = None
    demand: str | None = None
    avg_salary_usd: int | None = None


def load_job_roles(path: Path | None = None) -> list[JobRole]:
    path = path or (DATA_DIR / "job_roles.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    roles: list[JobRole] = []
    for item in raw:
        roles.append(
            JobRole(
                role=str(item["role"]),
                skills=[str(s) for s in item.get("skills", [])],
                description=item.get("description"),
                track=item.get("track"),
                demand=item.get("demand"),
                avg_salary_usd=item.get("avg_salary_usd"),
            )
        )
    return roles


def load_courses(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "courses.csv")
    df = pd.read_csv(path)
    for col in ["skill", "title", "provider", "level"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df
