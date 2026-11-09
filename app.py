from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SkillGapAI", layout="wide")

from skillgap_analyzer.ui_app import run_app

run_app()
