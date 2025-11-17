import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Settings
# -------------------------------
st.set_page_config(
    page_title="AI Career & Salary Advisor",
    page_icon="🎯",
    layout="wide"
)

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("career_role_model.pkl")
    reg = joblib.load("salary_model.pkl")
    return clf, reg

clf_model, reg_model = load_models()

# -------------------------------
# Page Header
# -------------------------------
st.markdown("""
# 🎯 AI Career Recommendation & Salary Predictor  
### *Get your ideal career role + accurate salary range instantly.*
---
""")

# -------------------------------
# User Inputs
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    years_exp = st.number_input("Years of Experience", 0.0, 40.0, 2.0)
    num_skills = st.number_input("Number of Relevant Skills", 1, 50, 5)
    portfolio_score = st.slider("Portfolio Score (0–100)", 0, 100, 60)

with col2:
    cert_count = st.number_input("Number of Certifications", 0, 20, 1)
    projects = st.number_input("Projects Completed", 0, 100, 3)
    education = st.selectbox(
        "Highest Education",
        ["HighSchool", "Bachelors", "Masters", "PhD"]
    )

with col3:
    current_role = st.selectbox(
        "Current Role",
        ["Intern", "Junior Developer", "Developer",
         "Senior Developer", "Team Lead", "Manager"]
    )
    location_tier = st.selectbox("Location Tier", ["Tier1", "Tier2", "Tier3"])
    preferred_domain = st.selectbox(
        "Preferred Domain",
        ["Frontend", "Backend", "Fullstack", "Data", "DevOps", "Mobile", "QA"]
    )

submit = st.button("🚀 Predict Career & Salary")

# -------------------------------
# Prediction Logic
# -------------------------------
if submit:

    # Prepare input vector
    input_df = pd.DataFrame([{
        "years_exp": years_exp,
        "num_skills": num_skills,
        "portfolio_score": portfolio_score,
        "cert_count": cert_count,
        "projects": projects,
        "education": education,
        "current_role": current_role,
        "location_tier": location_tier
    }])

    # Predict Role
    role_pred = clf_model.predict(input_df)[0]

    # Predict Salary (model gives salary in THOUSANDS)
    salary_pred = reg_model.predict(input_df)[0]

    # FIX: convert salary from "thousands" → rupees
    salary_pred = salary_pred * 100  

    # Salary range (±10%)
    low = salary_pred * 0.9
    high = salary_pred * 1.1

    # --------------------------
    # Display Output
    # --------------------------
    st.success(f"### 🎓 Recommended Role: **{role_pred}**")
    st.info(f"### 💰 Estimated Salary Range: ₹{low:,.0f} - ₹{high:,.0f} / month")

    st.markdown("---")
    st.subheader("📈 Career Growth Projection (Next 5 Years)")

    years = np.arange(0, 6)
    growth = [salary_pred * (1 + 0.12) ** y for y in years]

    fig, ax = plt.subplots()
    ax.plot(years, growth, marker="o")
    ax.set_xlabel("Years Ahead")
    ax.set_ylabel("Estimated Salary (INR)")
    ax.grid(True)
    st.pyplot(fig)

    # --------------------------
    # Skill Gap Suggestions
    # --------------------------
    st.markdown("---")
    st.subheader("🛠 Skill Gap Analysis")

    suggestions = []

    if num_skills < 6:
        suggestions.append("Add 3–5 high-demand domain skills.")
    if portfolio_score < 70:
        suggestions.append("Improve your portfolio with 1–2 live projects.")
    if cert_count < 2:
        suggestions.append("Complete at least 2 industry certifications.")
    if years_exp < 2:
        suggestions.append("Gain experience via internships/freelancing.")

    if not suggestions:
        suggestions.append("Great profile! Start preparing for leadership roles.")

    for s in suggestions:
        st.write("✔", s)

    # --------------------------
    # Downloadable Report
    # --------------------------
    st.markdown("---")
    st.subheader("📄 Download Your Career Report")

    report_text = f"""
Career Recommendation & Salary Report
-------------------------------------

Recommended Role: {role_pred}
Salary Range: ₹{low:,.0f} - ₹{high:,.0f}

User Profile:
{input_df.to_string(index=False)}

Suggestions:
{ " | ".join(suggestions) }
"""

    st.download_button(
        label="Download Report (.txt)",
        data=report_text,
        file_name="career_report.txt",
        mime="text/plain"
    )
