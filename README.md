
An advanced and comprehensive Applicant Tracking System (ATS) analysis tool built with Streamlit and leveraging the Gemini API for qualitative, AI-driven candidate assessment. This application goes beyond simple keyword matching to provide a holistic score for resume fit against a specific job description.

✨ Key Features
Multi-File Support: Upload one or more resumes simultaneously (PDF and DOCX formats supported).
Comprehensive Weighted Scoring: Calculates a final ATS score based on five key metrics:
Technical Skill Match (40%): Exact and fuzzy matching of industry-standard technical skills.
Contextual Keyword Match (20%): Alignment with general job description terminology.
Estimated Experience (20%): Heuristic calculation of total career tenure.
Role Title Alignment (10%): Match against the primary job role defined in the JD.
AI Critical Fit (10%): New! Qualitative assessment of soft skills, clarity, and overall professional presentation by the Gemini model.
AI Critical Assessment: Integrates the Gemini API to generate a qualitative "AI Fit Score" and a narrative summary for the top candidate, providing insights into communication style and soft skill alignment.
Interactive Visuals: Features a detailed Radar Chart visualizing the candidate's performance across the four quantitative scoring dimensions and the AI Fit score.
Candidate Ranking: Automatically sorts all uploaded resumes by the final ATS score and presents them in a clean scorecard table.
Data Export: Download the full results (scores, metrics, and filenames) as a CSV file.

Install Dependencies:
pip install -r requirements.txt
streamlit run app.py
