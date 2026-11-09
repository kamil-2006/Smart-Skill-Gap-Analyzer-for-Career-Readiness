from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from spacy.matcher import PhraseMatcher

from .config import ROOT_DIR
from .skills import default_skill_ontology, extract_skills
from .text import normalize_text


PROFILES_PATH = ROOT_DIR / ".local" / "user_profiles.json"


@dataclass(frozen=True)
class ResumeProfileExtraction:
    skills: set[str]
    education: list[str]
    certifications: list[str]
    experience_years: int | None


@dataclass(frozen=True)
class UserProfile:
    profile_id: str
    display_name: str
    full_name: str = ""
    email: str = ""
    phone: str = ""
    target_role: str = ""
    skills: list[str] = field(default_factory=list)
    skill_levels: dict[str, int] = field(default_factory=dict)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    experience_years: int = 0
    resume_filename: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @staticmethod
    def from_dict(raw: dict) -> UserProfile:
        return UserProfile(
            profile_id=str(raw.get("profile_id", "")).strip() or "default",
            display_name=str(raw.get("display_name", "")).strip() or "Default Profile",
            full_name=str(raw.get("full_name", "")).strip(),
            email=str(raw.get("email", "")).strip(),
            phone=str(raw.get("phone", "")).strip(),
            target_role=str(raw.get("target_role", "")).strip(),
            skills=[str(s).strip().lower() for s in raw.get("skills", []) if str(s).strip()],
            skill_levels={str(k).strip().lower(): int(v) for k, v in dict(raw.get("skill_levels", {})).items()},
            education=[str(s).strip() for s in raw.get("education", []) if str(s).strip()],
            certifications=[str(s).strip() for s in raw.get("certifications", []) if str(s).strip()],
            experience_years=max(0, int(raw.get("experience_years", 0) or 0)),
            resume_filename=(str(raw.get("resume_filename")).strip() if raw.get("resume_filename") else None),
            created_at=str(raw.get("created_at", "")).strip(),
            updated_at=str(raw.get("updated_at", "")).strip(),
        )

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "target_role": self.target_role,
            "skills": list(self.skills),
            "skill_levels": dict(self.skill_levels),
            "education": list(self.education),
            "certifications": list(self.certifications),
            "experience_years": int(self.experience_years),
            "resume_filename": self.resume_filename,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        val = str(item).strip()
        if not val:
            continue
        key = normalize_text(val)
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def make_profile_id(display_name: str) -> str:
    base = normalize_text(display_name or "").replace(" ", "-")
    base = re.sub(r"[^a-z0-9\-]", "", base).strip("-")
    return base or "profile"


def _read_profiles(path: Path | None = None) -> dict[str, UserProfile]:
    target = path or PROFILES_PATH
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    out: dict[str, UserProfile] = {}
    for item in raw.get("profiles", []):
        p = UserProfile.from_dict(item)
        out[p.profile_id] = p
    return out


def _write_profiles(profiles: dict[str, UserProfile], path: Path | None = None) -> None:
    target = path or PROFILES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profiles": [p.to_dict() for p in sorted(profiles.values(), key=lambda x: x.display_name.lower())]
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def list_profiles(path: Path | None = None) -> list[UserProfile]:
    profiles = _read_profiles(path)
    return sorted(profiles.values(), key=lambda x: x.display_name.lower())


def get_profile(profile_id: str, path: Path | None = None) -> UserProfile | None:
    return _read_profiles(path).get(profile_id)


def upsert_profile(profile: UserProfile, path: Path | None = None) -> UserProfile:
    profiles = _read_profiles(path)
    now = _utc_now_iso()
    existing = profiles.get(profile.profile_id)
    created = existing.created_at if existing and existing.created_at else (profile.created_at or now)
    cleaned = UserProfile(
        profile_id=profile.profile_id,
        display_name=profile.display_name.strip() or "Untitled Profile",
        full_name=profile.full_name.strip(),
        email=profile.email.strip(),
        phone=profile.phone.strip(),
        target_role=profile.target_role.strip(),
        skills=sorted({str(s).strip().lower() for s in profile.skills if str(s).strip()}),
        skill_levels={str(k).strip().lower(): int(v) for k, v in profile.skill_levels.items()},
        education=_dedupe_keep_order(profile.education),
        certifications=_dedupe_keep_order(profile.certifications),
        experience_years=max(0, int(profile.experience_years)),
        resume_filename=profile.resume_filename,
        created_at=created,
        updated_at=now,
    )
    profiles[cleaned.profile_id] = cleaned
    _write_profiles(profiles, path)
    return cleaned


def create_profile(display_name: str, path: Path | None = None) -> UserProfile:
    profiles = _read_profiles(path)
    base = make_profile_id(display_name)
    candidate = base
    suffix = 2
    while candidate in profiles:
        candidate = f"{base}-{suffix}"
        suffix += 1

    now = _utc_now_iso()
    profile = UserProfile(
        profile_id=candidate,
        display_name=(display_name.strip() or "New Profile"),
        created_at=now,
        updated_at=now,
    )
    profiles[candidate] = profile
    _write_profiles(profiles, path)
    return profile


@lru_cache(maxsize=1)
def _spacy_nlp():
    import spacy

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _nltk_tokens(text: str) -> list[str]:
    try:
        from nltk.tokenize import wordpunct_tokenize

        return wordpunct_tokenize(text)
    except Exception:
        return re.findall(r"[A-Za-z0-9\+\#\.]+", text)


def _nltk_sentences(text: str) -> list[str]:
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()]


def _line_candidates(text: str) -> list[str]:
    lines = [re.sub(r"^[\s\-\*\u2022]+", "", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _skill_extraction_nltk_spacy(text: str) -> set[str]:
    ontology = default_skill_ontology()
    found, _ = extract_skills(text, ontology=ontology)

    normalized = normalize_text(text)
    tokens = [normalize_text(tok) for tok in _nltk_tokens(normalized)]
    tokens = [t for t in tokens if t]

    ngram_set: set[str] = set(tokens)
    for n in (2, 3):
        for i in range(len(tokens) - n + 1):
            ngram_set.add(" ".join(tokens[i : i + n]))

    for canonical, forms in ontology.items():
        forms_norm = {normalize_text(f) for f in forms}
        if forms_norm & ngram_set:
            found.add(canonical)

    nlp = _spacy_nlp()
    doc = nlp(normalized)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    form_to_canonical: dict[str, str] = {}
    patterns = []
    for canonical, forms in ontology.items():
        for form in forms:
            key = normalize_text(form)
            if not key:
                continue
            form_to_canonical[key] = canonical
            patterns.append(nlp.make_doc(key))

    matcher.add("SKILL_TERMS", patterns)
    for _match_id, start, end in matcher(doc):
        key = doc[start:end].text.lower().strip()
        canonical = form_to_canonical.get(key)
        if canonical:
            found.add(canonical)
    return found


def _extract_education(text: str) -> list[str]:
    keywords = {
        "education",
        "bachelor",
        "master",
        "phd",
        "b.tech",
        "b tech",
        "m.tech",
        "m tech",
        "b.e",
        "m.e",
        "b.sc",
        "m.sc",
        "diploma",
        "university",
        "college",
        "cgpa",
    }

    candidates: list[str] = []
    for line in _line_candidates(text):
        low = normalize_text(line)
        if any(k in low for k in keywords):
            candidates.append(line)

    for sent in _nltk_sentences(text):
        low = normalize_text(sent)
        if any(k in low for k in keywords):
            candidates.append(sent.strip())

    nlp = _spacy_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms = [
        "bachelor of technology",
        "master of technology",
        "bachelor of science",
        "master of science",
        "doctor of philosophy",
        "btech",
        "mtech",
        "b.e",
        "m.e",
    ]
    matcher.add("EDU_TERMS", [nlp.make_doc(t) for t in terms])
    doc = nlp(text)
    for _match_id, start, end in matcher(doc):
        span = doc[max(0, start - 5) : min(len(doc), end + 12)]
        candidates.append(span.text.strip())

    return _dedupe_keep_order(candidates)[:8]


def _extract_certifications(text: str) -> list[str]:
    keywords = {
        "certification",
        "certifications",
        "certified",
        "certificate",
        "course completion",
        "credential",
    }

    lines = _line_candidates(text)
    candidates: list[str] = []
    for line in lines:
        low = normalize_text(line)
        if any(k in low for k in keywords):
            candidates.append(line)

    nlp = _spacy_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    terms = [
        "aws certified",
        "azure fundamentals",
        "google cloud",
        "pmp",
        "scrum master",
        "oracle certified",
        "coursera certificate",
        "udemy certificate",
    ]
    matcher.add("CERT_TERMS", [nlp.make_doc(t) for t in terms])
    doc = nlp(text)
    for _match_id, start, end in matcher(doc):
        span = doc[max(0, start - 4) : min(len(doc), end + 10)]
        candidates.append(span.text.strip())

    return _dedupe_keep_order(candidates)[:10]


def _extract_experience_years(text: str) -> int | None:
    normalized = normalize_text(text)
    years: list[float] = []

    for m in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b", normalized):
        try:
            years.append(float(m))
        except ValueError:
            pass

    if years:
        best = max(years)
        return max(0, int(round(best)))

    # Fallback: if resume explicitly says "experience" but no count, treat as unknown.
    if "experience" in normalized:
        return 0
    return None


def extract_profile_from_resume(text: str) -> ResumeProfileExtraction:
    skills = _skill_extraction_nltk_spacy(text)
    education = _extract_education(text)
    certifications = _extract_certifications(text)
    experience_years = _extract_experience_years(text)
    return ResumeProfileExtraction(
        skills=skills,
        education=education,
        certifications=certifications,
        experience_years=experience_years,
    )
