from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from skillgap_analyzer.data import load_courses, load_job_roles
from skillgap_analyzer.gap import analyze_gap
from skillgap_analyzer.models import load_or_train, predict
from skillgap_analyzer.recommender import recommend_courses
from skillgap_analyzer.resume_parser import parse_upload
from skillgap_analyzer.skills import extract_skills, skills_to_text
from skillgap_analyzer.viz import make_charts


st.set_page_config(page_title="Smart Skill Gap Analyzer", layout="wide")


@st.cache_data
def _job_roles():
    return load_job_roles()


@st.cache_data
def _courses_df() -> pd.DataFrame:
    return load_courses()


@st.cache_resource
def _models():
    return load_or_train()


st.title("Smart Skill Gap Analyzer for Career Readiness")
st.caption("Resume skill extraction • Skill gap analysis • Readiness prediction • Learning recommendations")

roles = _job_roles()
role_names = [r.role for r in roles]
courses_df = _courses_df()

with st.sidebar:
    st.header("Inputs")
    resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    resume_text_manual = st.text_area("Or paste resume text", height=140, placeholder="Paste your resume text here…")

    st.subheader("Target job")
    target_mode = st.radio("Choose target input", ["Job role (dataset)", "Paste job description"], horizontal=False)
    if target_mode == "Job role (dataset)":
        selected_role = st.selectbox("Target role", role_names, index=0)
        job_description = ""
    else:
        selected_role = ""
        job_description = st.text_area("Job description", height=140, placeholder="Paste a job description here…")

    st.subheader("Learning effort")
    hours_per_week = int(st.slider("Learning hours / week", min_value=1, max_value=30, value=10))
    experience_years = int(st.slider("Experience (years)", min_value=0, max_value=10, value=0))

    run = st.button("Analyze", type="primary", use_container_width=True)


def _get_target_skills() -> set[str]:
    if target_mode == "Job role (dataset)":
        role = next(r for r in roles if r.role == selected_role)
        return set(role.skills)
    skills, _ = extract_skills(job_description)
    return skills


def _get_resume_text() -> str:
    if resume_file is not None:
        parsed = parse_upload(resume_file.name, resume_file.read())
        return parsed.text
    return resume_text_manual or ""


if not run:
    st.info("Upload a resume (or paste text), choose a target role, then click Analyze.")
    st.stop()

resume_text = _get_resume_text()
if not resume_text.strip():
    st.error("No resume text found. Upload a resume file or paste resume text.")
    st.stop()

user_skills, user_matches = extract_skills(resume_text)
target_skills = _get_target_skills()

if not target_skills:
    st.error("No target skills found. Select a dataset role or paste a job description with recognizable skills.")
    st.stop()

gap = analyze_gap(user_skills, target_skills)
bundle = _models()
preds = predict(
    bundle,
    coverage=gap.coverage,
    missing_count=len(gap.missing),
    hours_per_week=hours_per_week,
    experience_years=experience_years,
)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage", f"{gap.coverage*100:.0f}%")
    m2.metric("Cosine similarity", f"{gap.cosine:.2f}")
    m3.metric("Missing skills", str(len(gap.missing)))
    m4.metric("Matched skills", str(len(gap.matched)))

    st.subheader("Skills")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Extracted from resume**")
        st.code(skills_to_text(user_skills) or "—", language="text")
    with c2:
        st.markdown("**Target skills**")
        st.code(skills_to_text(target_skills) or "—", language="text")

    c3, c4, c5 = st.columns(3)
    with c3:
        st.markdown("**Missing**")
        st.code(skills_to_text(gap.missing) or "—", language="text")
    with c4:
        st.markdown("**Matched**")
        st.code(skills_to_text(gap.matched) or "—", language="text")
    with c5:
        st.markdown("**Extra**")
        st.code(skills_to_text(gap.extra) or "—", language="text")

with right:
    st.subheader("Predictions")
    readiness_pct = preds.ready_proba * 100.0
    st.metric("Job readiness probability", f"{readiness_pct:.0f}%")
    st.progress(min(1.0, max(0.0, preds.ready_proba)))

    st.markdown("**Time-to-ready (weeks)**")
    p1, p2 = st.columns(2)
    p1.metric("Linear Regression", f"{preds.time_weeks_lr:.1f}")
    p2.metric("Random Forest", f"{preds.time_weeks_rf:.1f}")

    st.caption("Synthetic training data is used by default; plug in real user progression logs when available.")

st.divider()

charts = make_charts(
    matched=gap.matched,
    missing=gap.missing,
    extra=gap.extra,
    predicted_weeks=float(preds.time_weeks_rf),
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Gap counts")
    st.plotly_chart(charts.coverage_bar, use_container_width=True)
with c2:
    st.subheader("Target coverage")
    st.plotly_chart(charts.skills_radar, use_container_width=True)
with c3:
    st.subheader("Progress curve")
    st.plotly_chart(charts.progression_line, use_container_width=True)

st.divider()

st.subheader("Learning recommendations")
recs = recommend_courses(courses_df, gap.missing, top_k_per_skill=3)
if recs.empty:
    st.write("No recommendations found for the missing skills in the local course catalog.")
else:
    recs = recs.copy()
    if "duration_hours" in recs.columns:
        recs["duration_hours"] = pd.to_numeric(recs["duration_hours"], errors="coerce")
    st.dataframe(recs, use_container_width=True, hide_index=True)

st.caption(
    "Tip: If your resume is image-scanned and skill extraction looks empty, convert it to text (selectable) or paste text directly."
)
