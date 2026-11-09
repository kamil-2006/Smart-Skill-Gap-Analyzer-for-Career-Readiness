# Smart Skill Gap Analyzer

A Streamlit app that compares resume skills with a target career, estimates readiness, and recommends learning resources.

## Features

- Extracts skills and profile information from PDF, DOCX, and TXT resumes.
- Compares a profile with built-in job roles or a pasted job description.
- Estimates time to readiness and readiness probability using locally trained models.
- Recommends courses from the included catalog.
- Saves profiles locally on your computer.

## Requirements

- Python 3.10 or newer
- pip

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

The app creates `.local/user_profiles.json` for profiles and saves trained model files under `models/`. These local files are excluded from Git. Do not add resumes or personal information to the repository.

## Project layout

- `app.py` — Streamlit entry point
- `skillgap_analyzer/` — app functionality
- `data/job_roles.json` — sample target roles
- `data/courses.csv` — recommendation catalog
- `data/user_profiles.json` — empty profile template
- `scripts/train_models.py` — optional model training

The research paper and its extracted text are intentionally excluded because they contain author-identifying information. The PDF extraction utility can be used with a PDF you supply:

```powershell
python extract_pdf.py --pdf path\to\document.pdf --out .local\extracted.txt
```
