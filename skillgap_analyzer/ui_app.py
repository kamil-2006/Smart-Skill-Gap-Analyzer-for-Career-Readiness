from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import JobRole, load_courses, load_job_roles
from .gap import analyze_gap
from .matching import rank_roles
from .models import load_or_train, predict
from .profile import (
    UserProfile,
    create_profile,
    extract_profile_from_resume,
    get_profile,
    list_profiles,
    upsert_profile,
)
from .recommender import recommend_courses
from .resume_parser import parse_upload
from .skills import extract_skills


CSS = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.stApp {
  background:
    radial-gradient(1200px 700px at 70% 0%, rgba(34,199,255,0.20), rgba(0,0,0,0) 60%),
    radial-gradient(900px 600px at 20% 15%, rgba(31,230,211,0.18), rgba(0,0,0,0) 55%),
    linear-gradient(180deg, #060B14 0%, #050814 100%);
  color: #E8F0FF;
}

h1, h2, h3 { letter-spacing: -0.02em; }
.sg-muted { color: rgba(232,240,255,0.68); }
.sg-pill {
  display: inline-flex;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  color: rgba(232,240,255,0.85);
  font-size: 12px;
  gap: 8px;
}
.sg-gradient {
  background: linear-gradient(90deg, #1FE6D3, #22C7FF);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.sg-card {
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 16px 46px rgba(0,0,0,0.45);
}
.sg-card.sel {
  border: 1px solid rgba(31,230,211,0.55);
  box-shadow: 0 16px 46px rgba(0,0,0,0.45), 0 0 0 1px rgba(31,230,211,0.18), 0 0 36px rgba(31,230,211,0.15);
}
.sg-card h4 { margin: 0 0 2px 0; }
.sg-card small { color: rgba(232,240,255,0.60); }

div.stButton > button {
  border: 0;
  border-radius: 14px;
  font-weight: 750;
  background: linear-gradient(90deg, #1FE6D3, #22C7FF);
  color: #021015;
  padding: 0.74rem 1.15rem;
  white-space: nowrap;
}
button[kind="secondary"] {
  background: rgba(255,255,255,0.07) !important;
  color: rgba(232,240,255,0.90) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
}
div.stButton > button:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}

div[data-baseweb="input"] > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}
div[data-baseweb="textarea"] textarea {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
}

div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }
</style>
"""


@st.cache_data
def _job_roles() -> list[JobRole]:
    return load_job_roles()


@st.cache_data
def _courses_df() -> pd.DataFrame:
    return load_courses()


@st.cache_resource
def _models():
    return load_or_train()


def _read_non_empty_lines(value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in (value or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _merge_text_lines(existing: list[str], incoming: list[str]) -> list[str]:
    return _read_non_empty_lines("\n".join(list(existing) + list(incoming)))


def _apply_profile_to_state(profile: UserProfile) -> None:
    st.session_state.active_profile_id = profile.profile_id
    st.session_state.profile_display_name = profile.display_name
    st.session_state.profile_full_name = profile.full_name
    st.session_state.profile_email = profile.email
    st.session_state.profile_phone = profile.phone
    st.session_state.profile_education = list(profile.education)
    st.session_state.profile_certifications = list(profile.certifications)
    st.session_state.profile_education_text = "\n".join(profile.education)
    st.session_state.profile_certifications_text = "\n".join(profile.certifications)
    st.session_state.selected_role = profile.target_role
    st.session_state.role_locked = bool(profile.target_role)
    st.session_state.skills = list(profile.skills)
    st.session_state.skill_levels = {str(k): int(v) for k, v in profile.skill_levels.items()}
    st.session_state.experience_years = int(profile.experience_years or 0)
    st.session_state.resume_loaded_name = profile.resume_filename


def _current_profile_snapshot() -> UserProfile:
    profile_id = str(st.session_state.get("active_profile_id") or "").strip() or "default"
    display_name = str(st.session_state.get("profile_display_name") or "").strip() or "Default Profile"
    education = _read_non_empty_lines(str(st.session_state.get("profile_education_text") or ""))
    certifications = _read_non_empty_lines(str(st.session_state.get("profile_certifications_text") or ""))
    if not education:
        education = list(st.session_state.get("profile_education", []))
    if not certifications:
        certifications = list(st.session_state.get("profile_certifications", []))
    st.session_state.profile_education = education
    st.session_state.profile_certifications = certifications
    return UserProfile(
        profile_id=profile_id,
        display_name=display_name,
        full_name=str(st.session_state.get("profile_full_name") or "").strip(),
        email=str(st.session_state.get("profile_email") or "").strip(),
        phone=str(st.session_state.get("profile_phone") or "").strip(),
        target_role=str(st.session_state.get("selected_role") or "").strip(),
        skills=[str(s).strip().lower() for s in st.session_state.get("skills", []) if str(s).strip()],
        skill_levels={str(k).strip().lower(): int(v) for k, v in dict(st.session_state.get("skill_levels", {})).items()},
        education=education,
        certifications=certifications,
        experience_years=max(0, int(st.session_state.get("experience_years", 0) or 0)),
        resume_filename=st.session_state.get("resume_loaded_name"),
    )


def _save_profile_snapshot() -> None:
    saved = upsert_profile(_current_profile_snapshot())
    st.session_state.profile_display_name = saved.display_name


def _bootstrap_profile_state() -> None:
    profiles = list_profiles()
    if not profiles:
        seeded = upsert_profile(UserProfile(profile_id="default", display_name="Default Profile"))
        _apply_profile_to_state(seeded)
        return

    preferred_id = str(st.session_state.get("active_profile_id") or "").strip()
    profile = get_profile(preferred_id) if preferred_id else None
    if profile is None:
        profile = profiles[0]
    _apply_profile_to_state(profile)


def _init_state(roles: list[JobRole]) -> None:
    st.session_state.setdefault("view", "home")  # home | profile | resume | role_search | skills | analysis | courses
    st.session_state.setdefault("selected_role", "")
    st.session_state.setdefault("role_locked", False)  # set True when user explicitly picks a role
    st.session_state.setdefault("role_query", "")
    st.session_state.setdefault("resume_text_manual", "")
    st.session_state.setdefault("resume_text_cache", "")
    st.session_state.setdefault("resume_analyzed", False)
    st.session_state.setdefault("skills", [])  # ordered canonical skills
    st.session_state.setdefault("skill_levels", {})  # canonical -> 0..100
    st.session_state.setdefault("hours_per_week", 10)
    st.session_state.setdefault("experience_years", 0)
    st.session_state.setdefault("resume_loaded_name", None)
    st.session_state.setdefault("active_profile_id", "default")
    st.session_state.setdefault("profile_display_name", "Default Profile")
    st.session_state.setdefault("profile_full_name", "")
    st.session_state.setdefault("profile_email", "")
    st.session_state.setdefault("profile_phone", "")
    st.session_state.setdefault("profile_education", [])
    st.session_state.setdefault("profile_certifications", [])
    st.session_state.setdefault("profile_education_text", "")
    st.session_state.setdefault("profile_certifications_text", "")
    st.session_state.setdefault("profile_manual_skills_input", "")
    st.session_state.setdefault("new_profile_name", "")
    st.session_state.setdefault("profile_initialized", False)
    st.session_state.setdefault("last_view_before_profile", "home")
    st.session_state.setdefault("pending_profile_load_id", "")
    st.session_state.setdefault("profile_panel_message", "")
    if st.session_state.view == "setup":
        st.session_state.view = "resume"
    if not st.session_state.profile_initialized:
        _bootstrap_profile_state()
        st.session_state.profile_initialized = True


def _pretty_skill(skill: str) -> str:
    s = skill.strip()
    if s.lower() in {"nlp", "sql", "aws", "gcp"}:
        return s.upper()
    if s.lower() in {"c++", "c#"}:
        return s
    if s.lower() == "node.js":
        return "Node.js"
    if s.lower() == "scikit-learn":
        return "Scikit-learn"
    return s.title()


def _canonicalize_skill(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    found, _ = extract_skills(text)
    if len(found) == 1:
        return next(iter(found))
    simplified = re.sub(r"\s+", " ", text).strip().lower()
    return simplified


def _merge_skills(new_skills: list[str], *, default_level: int = 60, autosave: bool = True) -> None:
    for s in new_skills:
        if s not in st.session_state.skills:
            st.session_state.skills.append(s)
        st.session_state.skill_levels.setdefault(s, int(default_level))
    if autosave:
        _save_profile_snapshot()


def _weighted_coverage(user_levels: dict[str, int], target: set[str]) -> float:
    if not target:
        return 0.0
    total = 0.0
    for skill in target:
        total += max(0, min(100, int(user_levels.get(skill, 0)))) / 100.0
    return float(total / len(target))


def _format_salary_inr(avg_salary_usd: int | None) -> str:
    if not avg_salary_usd:
        return "-"
    usd_to_inr = 83.0
    salary_inr = float(avg_salary_usd) * usd_to_inr
    return f"₹{salary_inr:,.0f}"


def _category_scores(target: set[str], user_levels: dict[str, int]) -> dict[str, float]:
    soft = {"communication", "problem solving", "teamwork"}
    framework = {
        "react",
        "node.js",
        "express",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "tensorflow",
        "pytorch",
        "streamlit",
        "fastapi",
        "flask",
        "django",
    }
    buckets = {"Technical": set(), "Framework": set(), "Soft": set()}
    for s in target:
        if s in soft:
            buckets["Soft"].add(s)
        elif s in framework:
            buckets["Framework"].add(s)
        else:
            buckets["Technical"].add(s)

    out: dict[str, float] = {}
    for k, skills in buckets.items():
        out[k] = _weighted_coverage(user_levels, skills) if skills else 0.0
    return out


def _skill_comparison_fig(target: list[str], user_levels: dict[str, int]) -> go.Figure:
    required = [100 for _ in target]
    user = [max(0, min(100, int(user_levels.get(s, 0)))) for s in target]

    fig = go.Figure()
    fig.add_bar(name="You", x=[_pretty_skill(s) for s in target], y=user, marker_color="#1FE6D3")
    fig.add_bar(name="Required", x=[_pretty_skill(s) for s in target], y=required, marker_color="#6D3CFF")
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(232,240,255,0.88)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def _readiness_gauge(value_pct: float) -> go.Figure:
    value_pct = float(max(0.0, min(100.0, value_pct)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_pct,
            number={"suffix": "%", "font": {"size": 44, "color": "rgba(232,240,255,0.92)"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
                "bar": {"color": "#1FE6D3"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": "rgba(255,255,255,0.08)"}],
            },
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(232,240,255,0.88)"),
    )
    return fig


def _chips(skills: list[str], *, tone: str = "neutral") -> str:
    if not skills:
        return '<span class="sg-muted">-</span>'
    bg = "rgba(255,255,255,0.06)" if tone == "neutral" else "rgba(255,80,80,0.10)"
    border = "rgba(255,255,255,0.10)" if tone == "neutral" else "rgba(255,80,80,0.25)"
    color = "rgba(232,240,255,0.86)" if tone == "neutral" else "rgba(255,130,130,0.95)"
    html = []
    for s in skills:
        html.append(
            f'<span class="sg-pill" style="background:{bg};border-color:{border};color:{color}; margin: 0 8px 8px 0;">'
            f"{_pretty_skill(s)}"
            "</span>"
        )
    return "".join(html)


def _logo_bar() -> None:
    left, right = st.columns([0.72, 0.28])
    with left:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom: 10px;">
              <div style="width:34px; height:34px; border-radius:12px; background: rgba(31,230,211,0.12);
                          border: 1px solid rgba(31,230,211,0.35); display:flex; align-items:center; justify-content:center;">
                <span style="font-size:12px; font-weight: 900;">SG</span>
              </div>
              <div style="font-weight:850; font-size:18px;">
                SkillGap<span class="sg-gradient">AI</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        current_view = str(st.session_state.get("view", "home"))
        profile_label = str(st.session_state.get("profile_display_name") or "Profile")
        st.markdown(
            f'<div class="sg-muted" style="text-align:right; font-size:12px; margin-top: 2px; margin-bottom: 6px;">{profile_label}</div>',
            unsafe_allow_html=True,
        )
        if current_view == "profile":
            st.button("User Profile", type="secondary", use_container_width=True, disabled=True, key="profile_btn_disabled")
        else:
            if st.button(
                "User Profile",
                type="secondary",
                use_container_width=True,
                key=f"profile_btn_{current_view}",
            ):
                _save_profile_snapshot()
                st.session_state.last_view_before_profile = current_view
                st.session_state.view = "profile"
                st.rerun()


def _metric_card(icon: str, label: str, value: str) -> None:
    icon_html = ""
    gap = "0px"
    if str(icon).strip():
        gap = "10px"
        icon_html = (
            '<div style="width:34px; height:34px; border-radius:12px; background: rgba(255,255,255,0.06);'
            ' border: 1px solid rgba(255,255,255,0.12); display:flex; align-items:center; justify-content:center;">'
            f'<span style="font-size:16px;">{icon}</span>'
            "</div>"
        )

    html = (
        '<div class="sg-card" style="padding: 14px 16px;">'
        f'<div style="display:flex; align-items:center; gap:{gap};">'
        f"{icon_html}"
        "<div>"
        f'<div style="font-size: 26px; font-weight: 900; line-height: 1.05;">{value}</div>'
        f'<div class="sg-muted" style="font-size: 12px;">{label}</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _flow_stepper(current_step: int) -> None:
    steps = ["Resume", "Job Search", "Skills", "Analysis", "Courses"]
    pills: list[str] = []
    for idx, label in enumerate(steps, start=1):
        if idx == current_step:
            pills.append(
                f'<span class="sg-pill" style="background: rgba(31,230,211,0.18); border-color: rgba(31,230,211,0.45);">{idx}. {label}</span>'
            )
        else:
            pills.append(f'<span class="sg-pill">{idx}. {label}</span>')
    st.markdown(f"<div style='display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 12px 0;'>{''.join(pills)}</div>", unsafe_allow_html=True)


def _home_view() -> None:
    _logo_bar()
    st.markdown(
        """
        <div style="text-align:center; margin-top: 34px; margin-bottom: 18px;">
          <div class="sg-pill" style="margin: 0 auto 14px auto; width: fit-content;">AI-Powered Career Intelligence</div>
          <div style="font-size: 64px; font-weight: 950; line-height: 1.02;">
            Close Your <span class="sg-gradient">Skill Gaps</span><br/>Faster Than Ever
          </div>
          <div class="sg-muted" style="font-size: 18px; margin-top: 12px;">
            Analyze your skills against job market demands, get AI-powered recommendations, and build a roadmap.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.view = "resume"
            st.rerun()

    st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            """
            <div class="sg-card">
              <div class="sg-pill" style="width: fit-content; margin-bottom: 10px;">Analysis</div>
              <h4>Deep Analysis</h4>
              <small>Compare your skills against role requirements.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            """
            <div class="sg-card">
              <div class="sg-pill" style="width: fit-content; margin-bottom: 10px;">Matching</div>
              <h4>Smart Matching</h4>
              <small>Readiness prediction and time-to-ready estimates.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            """
            <div class="sg-card">
              <div class="sg-pill" style="width: fit-content; margin-bottom: 10px;">Roadmap</div>
              <h4>AI Advisor</h4>
              <small>Plan learning steps and track progress.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _select_role(role_name: str) -> None:
    st.session_state.selected_role = role_name
    st.session_state.role_locked = True
    _save_profile_snapshot()


def _profile_setup_panel(roles: list[JobRole]) -> None:
    pending_profile_id = str(st.session_state.pop("pending_profile_load_id", "") or "").strip()
    if pending_profile_id:
        loaded_pending = get_profile(pending_profile_id)
        if loaded_pending is not None:
            _apply_profile_to_state(loaded_pending)

    panel_message = str(st.session_state.pop("profile_panel_message", "") or "").strip()
    if panel_message:
        st.success(panel_message)

    st.markdown(
        """
        <div class="sg-card" style="margin-bottom: 14px;">
          <h4 style="margin:0 0 6px 0;">User Profile Setup</h4>
          <small>Create and manage an individual profile used as the base for gap analysis and recommendations.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    profiles = list_profiles()
    if not profiles:
        _bootstrap_profile_state()
        profiles = list_profiles()

    by_id = {p.profile_id: p for p in profiles}
    options = list(by_id.keys())
    current_id = st.session_state.active_profile_id if st.session_state.active_profile_id in by_id else options[0]
    picked_id = st.selectbox(
        "Active profile",
        options=options,
        index=options.index(current_id),
        format_func=lambda pid: by_id[pid].display_name,
    )
    if picked_id != st.session_state.active_profile_id:
        st.session_state.pending_profile_load_id = picked_id
        st.rerun()

    p1, p2 = st.columns([0.75, 0.25])
    with p1:
        st.text_input(
            "Create a new profile",
            key="new_profile_name",
            placeholder="Example: Data Science Profile",
        )
    with p2:
        if st.button("Create Profile", type="secondary", use_container_width=True):
            name = str(st.session_state.get("new_profile_name") or "").strip()
            if not name:
                st.warning("Enter a profile name before creating a new profile.")
            else:
                created = create_profile(name)
                st.session_state.pending_profile_load_id = created.profile_id
                st.session_state.profile_panel_message = f"Created profile: {created.display_name}"
                st.rerun()

    i1, i2, i3 = st.columns(3)
    with i1:
        st.text_input("Full name", key="profile_full_name")
    with i2:
        st.text_input("Email", key="profile_email")
    with i3:
        st.text_input("Phone", key="profile_phone")

    role_names = [r.role for r in roles]
    role_options = [""] + role_names
    selected = st.session_state.selected_role if st.session_state.selected_role in role_names else ""
    role_pick = st.selectbox(
        "Target job role for this profile",
        options=role_options,
        index=role_options.index(selected),
        format_func=lambda x: "Select target role..." if x == "" else x,
    )
    if role_pick != st.session_state.selected_role:
        st.session_state.selected_role = role_pick
        st.session_state.role_locked = bool(role_pick)
        _save_profile_snapshot()

    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        st.text_input(
            "Manual skills update (comma separated)",
            key="profile_manual_skills_input",
            placeholder="python, sql, machine learning",
        )
    with c2:
        if st.button("Add Skills", type="secondary", use_container_width=True):
            raw = str(st.session_state.get("profile_manual_skills_input") or "")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            canonical = [_canonicalize_skill(p) for p in parts]
            cleaned = [c for c in canonical if c]
            if cleaned:
                _merge_skills(cleaned, default_level=60, autosave=True)
                st.session_state.profile_panel_message = "Profile skills updated."
                st.rerun()

    st.text_area(
        "Education (one item per line)",
        key="profile_education_text",
        height=110,
        placeholder="B.Tech in Computer Science, XYZ University",
    )
    st.text_area(
        "Certifications (one item per line)",
        key="profile_certifications_text",
        height=110,
        placeholder="AWS Certified Cloud Practitioner",
    )

    s1, s2 = st.columns([0.5, 0.5])
    with s1:
        if st.button("Save Profile Setup", type="primary", use_container_width=True):
            st.session_state.profile_education = _read_non_empty_lines(st.session_state.profile_education_text)
            st.session_state.profile_certifications = _read_non_empty_lines(st.session_state.profile_certifications_text)
            _save_profile_snapshot()
            st.success("Profile saved.")
    with s2:
        if st.button("Refresh Profile", type="secondary", use_container_width=True):
            st.session_state.pending_profile_load_id = str(st.session_state.active_profile_id)
            st.session_state.profile_panel_message = "Reloaded latest saved profile."
            st.rerun()


def _profile_view(roles: list[JobRole]) -> None:
    _logo_bar()
    st.markdown(
        """
        <div style="margin: 6px 0 18px 0;">
          <div style="font-size: 40px; font-weight: 950;">User Profile</div>
          <div class="sg-muted">Manage your profile details, target role, manual skills, education, and certifications.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _profile_setup_panel(roles)

    prev = str(st.session_state.get("last_view_before_profile") or "home")
    if prev not in {"home", "resume", "role_search", "skills", "analysis", "courses"}:
        prev = "home"

    b1, b2, _ = st.columns([0.25, 0.28, 0.47])
    with b1:
        if st.button("Back", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = prev
            st.rerun()
    with b2:
        if st.button("Go to Resume", type="primary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "resume"
            st.rerun()


def _resume_view(roles: list[JobRole]) -> None:
    _logo_bar()
    _flow_stepper(1)
    st.markdown(
        """
        <div style="margin: 6px 0 18px 0;">
          <div style="font-size: 44px; font-weight: 950;">Resume Analysis</div>
          <div class="sg-muted">Upload resume (PDF/DOCX/TXT) and extract skills, education, certifications, and experience.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 0.9])
    with left:
        resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], key="resume_upload")
        st.text_area(
            "Or paste resume text",
            key="resume_text_manual",
            height=170,
            placeholder="Paste your resume text here...",
        )
        analyze_resume = st.button("Analyze Resume and Continue", type="primary", use_container_width=True)

        if analyze_resume:
            resume_text = ""
            loaded_name = None
            if resume_file is not None:
                parsed = parse_upload(resume_file.name, resume_file.read())
                resume_text = parsed.text
                loaded_name = resume_file.name
            else:
                resume_text = st.session_state.resume_text_manual or ""
                loaded_name = "Pasted text" if resume_text.strip() else None

            if not resume_text.strip():
                st.warning("Please upload a resume or paste resume text before analyzing.")
            else:
                extracted = extract_profile_from_resume(resume_text)
                _merge_skills(sorted(extracted.skills), default_level=60, autosave=False)
                st.session_state.profile_education = _merge_text_lines(
                    st.session_state.profile_education,
                    extracted.education,
                )
                st.session_state.profile_certifications = _merge_text_lines(
                    st.session_state.profile_certifications,
                    extracted.certifications,
                )
                st.session_state.profile_education_text = "\n".join(st.session_state.profile_education)
                st.session_state.profile_certifications_text = "\n".join(st.session_state.profile_certifications)
                if extracted.experience_years is not None:
                    st.session_state.experience_years = max(
                        int(st.session_state.experience_years),
                        int(extracted.experience_years),
                    )
                st.session_state.resume_text_cache = resume_text
                st.session_state.resume_loaded_name = loaded_name
                st.session_state.resume_analyzed = True

                if (not st.session_state.role_locked) and extracted.skills:
                    matches = rank_roles(extracted.skills, roles, top_k=1)
                    if matches and matches[0].coverage > 0:
                        st.session_state.selected_role = matches[0].role.role

                _save_profile_snapshot()
                st.session_state.view = "role_search"
                st.rerun()

        if st.session_state.resume_loaded_name:
            st.caption(f"Loaded: {st.session_state.resume_loaded_name}")

    with right:
        st.markdown(
            """
            <div class="sg-card">
              <h4 style="margin:0 0 6px 0;">What happens here?</h4>
              <small>1. Parse resume text<br/>2. Use NLTK + spaCy extraction<br/>3. Update profile fields + skills<br/>4. Suggest target role and continue</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.skills:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="sg-card">
                  <h4 style="margin:0 0 10px 0;">Extracted Skills ({len(st.session_state.skills)})</h4>
                  <div>{_chips(sorted(st.session_state.skills))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.session_state.profile_education:
            st.markdown(
                f"""
                <div class="sg-card" style="margin-top:10px;">
                  <h4 style="margin:0 0 8px 0;">Education</h4>
                  <small>{'<br/>'.join(st.session_state.profile_education[:5])}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.session_state.profile_certifications:
            st.markdown(
                f"""
                <div class="sg-card" style="margin-top:10px;">
                  <h4 style="margin:0 0 8px 0;">Certifications</h4>
                  <small>{'<br/>'.join(st.session_state.profile_certifications[:5])}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    nav1, nav2, _ = st.columns([0.22, 0.26, 0.52])
    with nav1:
        if st.button("Back Home", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "home"
            st.rerun()
    with nav2:
        if st.button("Job Search Step", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "role_search"
            st.rerun()


def _role_search_view(roles: list[JobRole]) -> None:
    _logo_bar()
    _flow_stepper(2)
    st.markdown(
        """
        <div style="margin: 6px 0 18px 0;">
          <div style="font-size: 40px; font-weight: 950;">Job Search and Role Selection</div>
          <div class="sg-muted">Pick the target role based on your resume profile. Then continue to skill-level tuning or quick analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.session_state.selected_role
    user_skills = set(st.session_state.skills)
    suggestions = rank_roles(user_skills, roles, top_k=6) if user_skills else []

    if (not selected) and (not st.session_state.role_locked) and suggestions and suggestions[0].coverage > 0:
        st.session_state.selected_role = suggestions[0].role.role
        selected = st.session_state.selected_role
        _save_profile_snapshot()

    if suggestions:
        st.markdown('<div class="sg-muted">Suggested from resume analysis</div>', unsafe_allow_html=True)
        grid = st.columns(3)
        for idx, m in enumerate(suggestions):
            role = m.role
            with grid[idx % 3]:
                is_sel = role.role == selected
                cls = "sg-card sel" if is_sel else "sg-card"
                track = role.track or "Career"
                demand = role.demand or "-"
                pct = int(round(m.coverage * 100))
                st.markdown(
                    f"""
                    <div class="{cls}">
                      <h4>{role.role}</h4>
                      <small>{track} - {demand} demand</small><br/>
                      <small><b>{pct}%</b> resume fit ({m.matched_count} matched skills)</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                label = "Selected" if is_sel else "Select"
                btn_type = "primary" if is_sel else "secondary"
                if st.button(label, key=f"suggest_{role.role}", type=btn_type, use_container_width=True):
                    _select_role(role.role)
                    st.rerun()
    else:
        st.markdown('<div class="sg-muted">No resume-based suggestions yet. Search manually below.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.session_state.role_query = st.text_input(
        "Search job role",
        value=st.session_state.role_query,
        placeholder="Type a role name (e.g., Data Scientist, DevOps Engineer)...",
    )
    q = (st.session_state.role_query or "").strip().lower()
    filtered = [r for r in roles if q in r.role.lower() or q in (r.track or "").lower()] if q else []
    if q and not filtered:
        st.info("No matching roles found. Try a different search term.")

    show_roles = filtered[:9] if filtered else []
    if q and show_roles:
        st.caption(f"Showing {len(show_roles)} of {len(filtered)} matches")

    if show_roles:
        grid2 = st.columns(3)
        for idx, role in enumerate(show_roles):
            with grid2[idx % 3]:
                is_sel = role.role == selected
                cls = "sg-card sel" if is_sel else "sg-card"
                track = role.track or "Career"
                demand = role.demand or "-"
                st.markdown(
                    f"""
                    <div class="{cls}">
                      <h4>{role.role}</h4>
                      <small>{track} - {demand} demand</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                label = "Selected" if is_sel else "Select"
                btn_type = "primary" if is_sel else "secondary"
                if st.button(label, key=f"pick_{role.role}", type=btn_type, use_container_width=True):
                    _select_role(role.role)
                    st.rerun()

    role_names = [r.role for r in roles]
    options = [""] + role_names
    current = selected if selected in role_names else ""
    picked = st.selectbox(
        "All roles (searchable)",
        options=options,
        index=options.index(current),
        key="role_all_select",
        format_func=lambda x: "Select a role..." if x == "" else x,
    )
    if picked and picked != selected:
        _select_role(picked)
        st.rerun()

    selected_role_obj = next((r for r in roles if r.role == st.session_state.selected_role), None)
    if selected_role_obj is not None and st.session_state.skills:
        target = set(selected_role_obj.skills)
        matched = sorted(user_skills & target)
        coverage = (len(matched) / len(target)) if target else 0.0
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sg-card">
              <h4 style="margin:0 0 6px 0;">Resume Fit Snapshot</h4>
              <small><b>{selected_role_obj.role}</b> - {coverage*100:.0f}% skill overlap from resume</small><br/>
              <small class="sg-muted">Matched skills: {len(matched)} of {len(target)}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    b1, b2, b3, _ = st.columns([0.2, 0.28, 0.2, 0.32])
    with b1:
        if st.button("Back Resume", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "resume"
            st.rerun()
    with b2:
        if st.button("Continue to Skills", type="primary", use_container_width=True):
            if not st.session_state.selected_role:
                st.warning("Select a target role to continue.")
            else:
                _save_profile_snapshot()
                st.session_state.view = "skills"
                st.rerun()
    with b3:
        if st.button("Quick Analysis", type="secondary", use_container_width=True):
            if not st.session_state.selected_role:
                st.warning("Select a target role first.")
            elif not st.session_state.skills:
                st.warning("Analyze your resume or add at least one skill first.")
            else:
                _save_profile_snapshot()
                st.session_state.view = "analysis"
                st.rerun()


def _skills_view(roles: list[JobRole]) -> None:
    _logo_bar()
    _flow_stepper(3)
    st.markdown(
        """
        <div style="margin: 6px 0 18px 0;">
          <div style="font-size: 40px; font-weight: 950;">Skills and Learning Effort</div>
          <div class="sg-muted">Tune your current skill levels, remove/add skills, then run analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_role = st.session_state.selected_role
    role = next((r for r in roles if r.role == selected_role), None)
    if role is None:
        st.warning("Choose a target role in Job Search first.")
        if st.button("Go to Job Search", type="primary"):
            st.session_state.view = "role_search"
            st.rerun()
        return

    st.markdown(
        f"""
        <div class="sg-card">
          <h4 style="margin:0 0 4px 0;">Selected Role: {role.role}</h4>
          <small>{role.track or "Career"} - Demand: {role.demand or "-"}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 0.95])
    with left:
        st.subheader(f"Your Skills ({len(st.session_state.skills)})")
        c1, c2 = st.columns([1, 0.34])
        with c1:
            new_skill = st.text_input(
                "Add a skill",
                placeholder="Type a skill and press Add...",
                label_visibility="collapsed",
            )
        with c2:
            add = st.button("Add Skill", type="secondary", use_container_width=True)

        if add:
            canonical = _canonicalize_skill(new_skill)
            if canonical:
                _merge_skills([canonical], default_level=60)
                st.rerun()

        tools = st.columns([0.44, 0.44, 0.12])
        with tools[0]:
            if st.session_state.resume_text_cache and st.button(
                "Extract From Resume",
                type="secondary",
                use_container_width=True,
            ):
                extracted = extract_profile_from_resume(st.session_state.resume_text_cache)
                _merge_skills(sorted(extracted.skills), default_level=60, autosave=False)
                st.session_state.profile_education = _merge_text_lines(
                    st.session_state.profile_education,
                    extracted.education,
                )
                st.session_state.profile_certifications = _merge_text_lines(
                    st.session_state.profile_certifications,
                    extracted.certifications,
                )
                st.session_state.profile_education_text = "\n".join(st.session_state.profile_education)
                st.session_state.profile_certifications_text = "\n".join(st.session_state.profile_certifications)
                if extracted.experience_years is not None:
                    st.session_state.experience_years = max(
                        int(st.session_state.experience_years),
                        int(extracted.experience_years),
                    )
                _save_profile_snapshot()
                st.rerun()
        with tools[1]:
            if st.session_state.skills and st.button("Clear All Skills", type="secondary", use_container_width=True):
                st.session_state.skills = []
                st.session_state.skill_levels = {}
                _save_profile_snapshot()
                st.rerun()

        if not st.session_state.skills:
            st.markdown('<div class="sg-muted">No skills yet. Add manually or extract from resume.</div>', unsafe_allow_html=True)
        else:
            for s in list(st.session_state.skills):
                lvl = int(st.session_state.skill_levels.get(s, 60))
                row = st.columns([0.34, 0.50, 0.16])
                with row[0]:
                    st.markdown(f"**{_pretty_skill(s)}**")
                with row[1]:
                    st.session_state.skill_levels[s] = int(
                        st.slider(
                            f"Level - {_pretty_skill(s)}",
                            0,
                            100,
                            value=lvl,
                            key=f"lvl_{s}",
                            label_visibility="collapsed",
                        )
                    )
                    st.caption(f"{st.session_state.skill_levels[s]}%")
                with row[2]:
                    if st.button("Remove Skill", key=f"rm_{s}", type="secondary", use_container_width=True):
                        st.session_state.skills = [x for x in st.session_state.skills if x != s]
                        st.session_state.skill_levels.pop(s, None)
                        _save_profile_snapshot()
                        st.rerun()

    with right:
        st.subheader("Learning Effort")
        st.session_state.hours_per_week = int(
            st.slider("Learning hours / week", 1, 30, value=int(st.session_state.hours_per_week))
        )
        st.session_state.experience_years = int(
            st.slider("Experience (years)", 0, 10, value=int(st.session_state.experience_years))
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Save Progress to Profile", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.success("Profile updated.")
        if st.button("Run Analysis", type="primary", use_container_width=True):
            if not st.session_state.skills:
                st.warning("Add or extract at least one skill first.")
            else:
                _save_profile_snapshot()
                st.session_state.view = "analysis"
                st.rerun()

        if st.button("Back to Job Search", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "role_search"
            st.rerun()

        st.markdown(
            """
            <div class="sg-card" style="margin-top: 16px;">
              <h4 style="margin:0 0 4px 0;">Tip</h4>
              <small>If resume extraction looks weak, add missing skills manually and set realistic levels before analysis.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _analysis_payload(roles: list[JobRole]):
    selected_role = st.session_state.selected_role
    role = next((r for r in roles if r.role == selected_role), None)
    if role is None:
        st.error("Select a job role first.")
        st.session_state.view = "role_search"
        return None

    target_skills = set(role.skills)
    user_levels = {k: int(v) for k, v in st.session_state.skill_levels.items()}
    user_skills = {k for k, v in user_levels.items() if int(v) > 0}

    gap = analyze_gap(user_skills, target_skills)
    coverage = _weighted_coverage(user_levels, target_skills)
    bundle = _models()
    preds = predict(
        bundle,
        coverage=coverage,
        missing_count=len(gap.missing),
        hours_per_week=int(st.session_state.hours_per_week),
        experience_years=int(st.session_state.experience_years),
    )
    return role, target_skills, user_levels, gap, coverage, preds


def _analysis_view(roles: list[JobRole]) -> None:
    _logo_bar()
    _flow_stepper(4)
    payload = _analysis_payload(roles)
    if payload is None:
        return

    role, target_skills, user_levels, gap, coverage, preds = payload

    top = st.columns([1.25, 0.75])
    with top[0]:
        demand = role.demand or "-"
        salary = _format_salary_inr(role.avg_salary_usd)
        st.markdown(
            f"""
            <div style="margin: 10px 0 6px 0;">
              <div style="font-size: 44px; font-weight: 950;">
                Analysis for <span class="sg-gradient">{role.role}</span>
              </div>
              <div class="sg-muted">Market demand: <b>{demand}</b> - Avg salary (INR): <b>{salary}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top[1]:
        if st.button("AI Advisor", type="primary", use_container_width=True):
            st.info("AI Advisor is a future-work feature in the paper. Add a chatbot module when ready.")

    cards = st.columns(4)
    skill_score_10 = int(round(coverage * 10))
    with cards[0]:
        _metric_card("", "Skill Score", f"{skill_score_10}")
    with cards[1]:
        _metric_card("", "Time to Ready", f"{preds.time_weeks_rf:.0f}w")
    with cards[2]:
        _metric_card("", "Confidence", f"{preds.ready_proba*100:.0f}%")
    with cards[3]:
        _metric_card("", "Strengths", f"{len(gap.matched)}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.45, 0.75])
    with left:
        st.markdown('<div class="sg-card"><h3 style="margin:0 0 10px 0;">Skill Comparison</h3></div>', unsafe_allow_html=True)
        ordered_target = list(role.skills)[:10]
        st.plotly_chart(_skill_comparison_fig(ordered_target, user_levels), use_container_width=True)

    with right:
        readiness_pct = preds.ready_proba * 100.0
        st.markdown('<div class="sg-card"><h3 style="margin:0 0 10px 0;">Job Readiness</h3></div>', unsafe_allow_html=True)
        st.plotly_chart(_readiness_gauge(readiness_pct), use_container_width=True)

        st.markdown("<div class='sg-card' style='margin-top: 12px;'><h4 style='margin:0 0 8px 0;'>Category Scores</h4></div>", unsafe_allow_html=True)
        cats = _category_scores(target_skills, user_levels)
        for name, v in cats.items():
            st.write(f"{name}")
            st.progress(min(1.0, max(0.0, float(v))))
            st.caption(f"{v*100:.0f}%")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sg-card">
          <h3 style="margin:0 0 10px 0;">Missing Skills ({len(gap.missing)})</h3>
          <div>{_chips(sorted(gap.missing), tone="danger")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav1, nav2, nav3, _ = st.columns([0.22, 0.28, 0.24, 0.26])
    with nav1:
        if st.button("Back to Skills", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "skills"
            st.rerun()
    with nav2:
        if st.button("Course Suggestions", type="primary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "courses"
            st.rerun()
    with nav3:
        if st.button("Change Role", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "role_search"
            st.rerun()

    st.caption("Analysis and course recommendation are now separated into different redirect steps.")


def _courses_view(roles: list[JobRole], courses_df: pd.DataFrame) -> None:
    _logo_bar()
    _flow_stepper(5)
    payload = _analysis_payload(roles)
    if payload is None:
        return

    role, _target_skills, _user_levels, gap, _coverage, _preds = payload
    st.markdown(
        f"""
        <div style="margin: 10px 0 8px 0;">
          <div style="font-size: 40px; font-weight: 950;">
            Course Suggestions for <span class="sg-gradient">{role.role}</span>
          </div>
          <div class="sg-muted">Based on resume + selected role analysis. Focused on {len(gap.missing)} missing skills.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sg-card">
          <h4 style="margin:0 0 10px 0;">Missing Skills</h4>
          <div>{_chips(sorted(gap.missing), tone="danger")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    recs = recommend_courses(courses_df, gap.missing, top_k_per_skill=3)
    if recs.empty:
        st.write("No recommendations found for the missing skills in the local course catalog.")
    else:
        recs = recs.copy()
        if "duration_hours" in recs.columns:
            recs["duration_hours"] = pd.to_numeric(recs["duration_hours"], errors="coerce")

        if "link" in recs.columns:
            for skill, group in recs.groupby("skill", dropna=False):
                skill_label = _pretty_skill(str(skill))
                st.markdown(f"**{skill_label}**")
                for _, row in group.iterrows():
                    title = str(row.get("title", "")).strip() or "Course"
                    url = str(row.get("link", "")).strip()
                    provider = str(row.get("provider", "")).strip()
                    level = str(row.get("level", "")).strip()
                    duration = row.get("duration_hours", None)
                    duration_text = ""
                    if duration is not None and pd.notna(duration):
                        try:
                            duration_text = f"{int(float(duration))}h"
                        except (TypeError, ValueError):
                            duration_text = str(duration)

                    meta = " - ".join([x for x in [provider, level, duration_text] if x])
                    if url:
                        st.markdown(f"- [{title}]({url})" + (f" ({meta})" if meta else ""))
                    else:
                        st.markdown(f"- {title}" + (f" ({meta})" if meta else ""))
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        else:
            st.dataframe(recs, use_container_width=True, hide_index=True)

    c1, c2, c3, _ = st.columns([0.2, 0.22, 0.2, 0.38])
    with c1:
        if st.button("Back Analysis", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "analysis"
            st.rerun()
    with c2:
        if st.button("Back Skills", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "skills"
            st.rerun()
    with c3:
        if st.button("Change Role", type="secondary", use_container_width=True):
            _save_profile_snapshot()
            st.session_state.view = "role_search"
            st.rerun()


def run_app() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    roles = _job_roles()
    courses_df = _courses_df()
    _init_state(roles)

    view = st.session_state.view
    if view == "home":
        _home_view()
    elif view == "profile":
        _profile_view(roles)
    elif view == "resume":
        _resume_view(roles)
    elif view == "role_search":
        _role_search_view(roles)
    elif view == "skills":
        _skills_view(roles)
    elif view == "analysis":
        _analysis_view(roles)
    elif view == "courses":
        _courses_view(roles, courses_df)
    else:
        st.session_state.view = "home"
        st.rerun()
