from __future__ import annotations

from dataclasses import dataclass

from .data import JobRole


@dataclass(frozen=True)
class RoleMatch:
    role: JobRole
    coverage: float
    matched_count: int
    missing_count: int
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]


def rank_roles(user_skills: set[str], roles: list[JobRole], *, top_k: int = 6) -> list[RoleMatch]:
    """Rank job roles based on simple skill overlap."""
    if not user_skills or not roles:
        return []

    user = {str(s).strip().lower() for s in user_skills if str(s).strip()}

    matches: list[RoleMatch] = []
    for role in roles:
        target = {str(s).strip().lower() for s in role.skills if str(s).strip()}
        if not target:
            continue

        matched = sorted(user & target)
        missing = sorted(target - user)
        coverage = len(matched) / len(target) if target else 0.0

        matches.append(
            RoleMatch(
                role=role,
                coverage=float(coverage),
                matched_count=len(matched),
                missing_count=len(missing),
                matched_skills=tuple(matched),
                missing_skills=tuple(missing),
            )
        )

    matches.sort(key=lambda m: (m.coverage, m.matched_count, -m.missing_count, m.role.role.lower()), reverse=True)
    return matches[: max(0, int(top_k))]
