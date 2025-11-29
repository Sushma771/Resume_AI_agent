# pro_ats_app.py
import os
import io
import re
import json
import base64
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import difflib
from datetime import datetime
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document
import plotly.graph_objects as go

# Optional imports (AI + PDF)
try:
    import openai
except:
    openai = None

try:
    from weasyprint import HTML
except:
    HTML = None

# --------------------------------------------------
#   ✓ GLOBAL CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PRO ATS Resume Analyzer",
    layout="wide",
    page_icon="📄"
)

APP_TITLE = "PRO ATS Resume Analyzer — Modern / Corporate / Dark"

# --------------------------------------------------
#   ✓ GLOBAL SKILL DICTIONARY
# --------------------------------------------------
GLOBAL_SKILLS = [
    "python","java","javascript","typescript","c++","c#","react","angular","vue","node",
    "express","django","flask","fastapi","html","css","sql","mysql","postgresql","mongodb",
    "redis","aws","gcp","azure","docker","kubernetes","git","github","gitlab","graphql",
    "rest api","pandas","numpy","tensorflow","keras","pytorch","scikit-learn","nlp",
    "opencv","machine learning","deep learning","microservices","terraform","ci/cd"
]

# Common synonyms
SYNONYMS = {
    "js": "javascript",
    "ml": "machine learning",
    "dl": "deep learning",
    "tf": "tensorflow",
    "sklearn": "scikit-learn"
}

# --------------------------------------------------
#   ✓ FILE PARSING
# --------------------------------------------------
def extract_text_from_pdf(raw_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
        return text
    except:
        return ""

def extract_text_from_docx(raw_bytes: bytes) -> str:
    try:
        doc = Document(io.BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(lines).strip().lower()

# --------------------------------------------------
#   ✓ REAL EXPERIENCE EXTRACTION (FIXED)
# --------------------------------------------------
def estimate_experience_years(resume_text: str) -> float:
    text = resume_text.lower()
    total_months = 0
    current_year = datetime.now().year

    # 1. Year ranges (2020-2022, 2019 to 2021)
    patterns = [
        r'(\d{4})\s*[-–]\s*(\d{4}|present)',
        r'(\d{4})\s+to\s+(\d{4}|present)'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for start, end in matches:
            start = int(start)
            end = current_year if end == "present" else int(end)
            if start <= end:
                total_months += (end - start) * 12

    # 2. Month ranges (Oct 2025 – Nov 2025)
    month_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
    }

    month_pattern = r'([a-z]{3,9})\s+\d{4}\s*[-–]\s*([a-z]{3,9})\s+\d{4}'
    matches = re.findall(month_pattern, text)
    for m1, m2 in matches:
        try:
            start_m = month_map[m1[:3]]
            end_m = month_map[m2[:3]]
            diff = max(1, end_m - start_m)
            total_months += diff
        except:
            pass

    years = round(total_months / 12, 2)
    return max(0, min(years, 20))  # cap 20

# --------------------------------------------------
#   ✓ SKILL EXTRACTION
# --------------------------------------------------
def extract_jd_skills(jd_text: str) -> List[str]:
    jd = jd_text.lower()
    found = []
    for s in GLOBAL_SKILLS:
        if s in jd and s not in found:
            found.append(s)
    return found

def resume_skills(resume_text: str) -> List[str]:
    r = resume_text.lower()
    found = []
    for s in GLOBAL_SKILLS:
        if re.search(rf'(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])', r):
            found.append(s)
    for key, full in SYNONYMS.items():
        if key in r:
            found.append(full)
    return sorted(set(found))

def fuzzy_skills(resume_text: str) -> List[str]:
    tokens = re.findall(r'\b[a-z0-9+-]+\b', resume_text.lower())
    collected = set()
    for token in tokens:
        if len(token) >= 3:
            match = difflib.get_close_matches(token, GLOBAL_SKILLS, n=1, cutoff=0.75)
            if match:
                collected.add(match[0])
    return sorted(collected)

# --------------------------------------------------
#   ✓ SCORING
# --------------------------------------------------
def compute_scores(resume_text: str, jd_text: str) -> Dict[str, Any]:
    jd = jd_text.lower()
    jd_skills = extract_jd_skills(jd_text)

    base_skills = resume_skills(resume_text)
    fuzz = fuzzy_skills(resume_text)
    all_skills = sorted(set(base_skills + fuzz))

    # Skill score
    if jd_skills:
        matched = [s for s in all_skills if s in jd_skills]
        skill_score = len(matched) / len(jd_skills)
    else:
        matched = []
        skill_score = min(len(all_skills) / 5, 1)

    # Keyword score
    jd_words = re.findall(r'\b[a-z]{4,}\b', jd)
    kw_matches = [w for w in jd_words if w in resume_text]
    keyword_score = len(kw_matches) / max(1, len(jd_words))

    # Experience score
    years = estimate_experience_years(resume_text)
    experience_score = min(years / 5, 1)

    # Title score
    title_keywords = jd_text.lower().split()
    title_score = 1 if any(w in resume_text for w in title_keywords[:5]) else 0.5

    # Weighted ATS score
    ats = (
        skill_score * 0.40 +
        keyword_score * 0.25 +
        experience_score * 0.20 +
        title_score * 0.15
    )
    ats = round(min(max(ats, 0), 1) * 100, 1)

    return {
        "ats_score": ats,
        "skill_score": round(skill_score, 3),
        "keyword_score": round(keyword_score, 3),
        "experience_score": round(experience_score, 3),
        "title_score": round(title_score, 3),
        "years_estimated": years,
        "jd_skills": jd_skills,
        "all_skills": all_skills,
        "matched_skills": matched,
        "matched_keywords": kw_matches
    }

# --------------------------------------------------
#   ✓ RADAR CHART
# --------------------------------------------------
def radar_chart(scores):
    labels = ["Skill", "Keyword", "Experience", "Title"]
    vals = [
        scores["skill_score"],
        scores["keyword_score"],
        scores["experience_score"],
        scores["title_score"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False
    )
    return fig

# --------------------------------------------------
#   ✓ UI STYLING MODES (A / B / C)
# --------------------------------------------------
style = st.selectbox("Choose UI Mode", ["A - Modern SaaS", "B - Minimal Corporate", "C - Premium Dark"])

if style.startswith("A"):
    st.markdown("""
    <style>
    body {background: linear-gradient(180deg, #eef2ff, #ffffff);}
    .card {background:white;padding:20px;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,0.08);}
    </style>
    """, unsafe_allow_html=True)

elif style.startswith("B"):
    st.markdown("""
    <style>
    body {background: #ffffff;}
    .card {background:#f8fafc;padding:20px;border-radius:10px;border:1px solid #eeeeee;}
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    body {background:#0b1220;color:#dbeafe;}
    .card {background:rgba(255,255,255,0.05);backdrop-filter:blur(6px);
           padding:20px;border-radius:12px;border:1px solid rgba(255,255,255,0.1);}
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------
#   ✓ APP HEADER
# --------------------------------------------------
st.title(APP_TITLE)
st.write("Upload resumes and analyze detailed ATS scoring, radar charts, skill matching, and more.")

uploaded_files = st.file_uploader("Upload PDF/DOCX resumes", accept_multiple_files=True, type=["pdf","docx"])
jd_text = st.text_area("Paste Job Description", height=180)

if st.button("Run ATS Analysis"):
    if not uploaded_files:
        st.error("Upload at least one resume.")
        st.stop()
    if not jd_text.strip():
        st.error("Paste a Job Description.")
        st.stop()

    results = []
    for f in uploaded_files:
        raw = f.read()
        if f.name.endswith(".pdf"):
            text = extract_text_from_pdf(raw)
        else:
            text = extract_text_from_docx(raw)

        normalized = normalize_text(text)
        scores = compute_scores(normalized, jd_text)
        scores["filename"] = f.name
        results.append(scores)

    # Sort by ATS
    df = pd.DataFrame(results).sort_values("ats_score", ascending=False)

    st.subheader("📊 Ranking")
    st.dataframe(df[["filename","ats_score","years_estimated","skill_score","keyword_score","experience_score"]])

    # Show best candidate
    top = results[0]

    st.markdown(f"<div class='card'><h2>{top['filename']} — {top['ats_score']}%</h2></div>", unsafe_allow_html=True)
    st.plotly_chart(radar_chart(top), use_container_width=True)

    st.write("### Estimated Experience:")
    st.write(f"**{top['years_estimated']} years (~ {round(top['years_estimated']*12)} months)**")

    st.write("### JD Skills:", top["jd_skills"])
    st.write("### Matched Skills:", top["matched_skills"])
    st.write("### All Detected Skills:", ", ".join(top["all_skills"]))

    # Download CSV
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        file_name="ats_results.csv"
    )
