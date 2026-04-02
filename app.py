import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, time, warnings, io, textwrap
from PIL import Image
import requests
import json
import plotly.express as px

# ── PDF generation ──────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# ★  HARDCODED API KEY  ★
#    Replace the string below with your Groq key
# ─────────────────────────────────────────────
GROQ_API_KEY = "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"   # ← paste your key here

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chronic Disease Forecasting",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.stApp { background: #f7f9fc; color: #1f2937; }
.main-header {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    padding: 22px 30px; border-radius: 16px; margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.main-header h1 { color: #ffffff; font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85); margin: 4px 0 0; }
.section-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 14px; padding: 20px; margin-bottom: 16px;
}
.kpi-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:20px; text-align:center; }
.kpi-value { font-size:2rem; font-weight:700; color:#111827; }
.kpi-label { font-size:0.8rem; color:#6b7280; }
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg,#4f46e5,#6366f1);
    color: white; border: none; border-radius: 10px; font-weight: 600;
}
div[data-testid="stButton"] > button:hover { opacity: 0.9; }
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #eef2ff; padding: 8px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] { border-radius:10px; color:#374151; font-weight:500; }
.stTabs [aria-selected="true"] { background:#4f46e5 !important; color:white !important; }
.risk-high   { background:#fee2e2; border-left:4px solid #ef4444; padding:12px; border-radius:8px; margin-bottom:8px; }
.risk-medium { background:#fef3c7; border-left:4px solid #f59e0b; padding:12px; border-radius:8px; margin-bottom:8px; }
.risk-low    { background:#dcfce7; border-left:4px solid #22c55e; padding:12px; border-radius:8px; margin-bottom:8px; }
.no-data-box {
    background: #f0f4ff; border: 2px dashed #a5b4fc;
    border-radius: 16px; padding: 60px 40px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "patient_df": None,
    "image_data": None,
    "forecast_result": "",
    "intervention_result": "",
    "risk_result": "",
    "prevention_plan": "",
    "ml_pred_html": "",
    "selected_patient_summary": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

TEMPERATURE = 0.3
MAX_TOKENS  = 2048

# ─────────────────────────────────────────────
# NO-DATA PLACEHOLDER HELPER
# ─────────────────────────────────────────────
def no_data_placeholder():
    st.markdown("""
    <div class="no-data-box">
      <h2>📂 No Data Loaded Yet</h2>
      <p style="color:#6b7280; font-size:1.05rem; margin-top:10px;">
        Please go to the <b>📂 Data Input</b> tab and either upload your patient
        data file or click <b>Load Sample Dataset</b> to continue.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GROQ LLM HELPER  (uses hardcoded key)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert clinical AI assistant specializing in chronic disease epidemiology,
predictive modeling, and evidence-based medicine. Your responses must be:
- Clinically accurate and grounded in medical literature
- Structured clearly with numbered lists, sections, and bullet points
- Risk-stratified (High / Medium / Low labels where applicable)
- Specific, actionable, and quantitative wherever possible
- Include confidence levels (High/Moderate/Low) with each prediction
- Cite relevant biomarkers, lab values, and clinical thresholds
Always include a disclaimer that predictions are AI-assisted and require clinical validation."""

def call_groq(user_message: str) -> str:
    api_key = GROQ_API_KEY
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        return "⚠️ API key not set. Please replace `YOUR_GROQ_API_KEY_HERE` in the source code with your actual Groq API key."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "top_p": 0.9,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return f"API Error {resp.status_code}: {resp.text}"
        except Exception as e:
            if attempt == 2:
                return f"Connection error: {str(e)}"
            time.sleep(1)
    return "Max retries exceeded. Please try again."

# ─────────────────────────────────────────────
# PDF GENERATION HELPERS
# ─────────────────────────────────────────────
def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Title2", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#4f46e5"),
        spaceAfter=6, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="SectionHead", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=14, spaceAfter=4,
        borderPad=4
    ))
    styles.add(ParagraphStyle(
        name="Body2", parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, leading=12,
        spaceBefore=12
    ))
    return styles

def _clean_md(text: str) -> str:
    """Strip basic markdown so ReportLab doesn't choke."""
    import re
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",   r"\1", text)
    text = re.sub(r"#{1,6}\s*",   "",    text)
    text = re.sub(r"`{1,3}",      "",    text)
    return text

def build_report_pdf(title: str, sections: dict, patient_data: dict = None) -> bytes:
    """
    sections = {"Section Name": "text content", ...}
    patient_data = dict of patient fields (optional header table)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = _pdf_styles()
    story  = []

    # ── Header banner ────────────────────────────────────────────────────
    story.append(Paragraph("🏥 Chronic Disease Forecasting Platform", styles["Title2"]))
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4f46e5")))
    story.append(Spacer(1, 10))

    # ── Patient data table ───────────────────────────────────────────────
    if patient_data:
        story.append(Paragraph("Patient Profile", styles["SectionHead"]))
        rows = [["Field", "Value"]]
        for k, v in patient_data.items():
            rows.append([str(k).replace("_", " "), str(v)])
        t = Table(rows, colWidths=[2.2*inch, 4.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#4f46e5")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
            ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#d1d5db")),
            ("PADDING",     (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

    # ── Content sections ─────────────────────────────────────────────────
    for sec_title, content in sections.items():
        story.append(Paragraph(sec_title, styles["SectionHead"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
        story.append(Spacer(1, 4))
        if content:
            for line in _clean_md(content).split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(line, styles["Body2"]))
        story.append(Spacer(1, 8))

    # ── Disclaimer ───────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "DISCLAIMER: All AI-generated predictions are for clinical decision support only "
        "and must be validated by a qualified healthcare professional before clinical use. "
        "Generated by Chronic Disease Forecasting Platform · Powered by Groq / Llama 3.3.",
        styles["Disclaimer"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()

def build_patient_summary_pdf(patient_data: dict,
                               forecast: str, intervention: str,
                               risk_analysis: str, prevention: str) -> bytes:
    sections = {
        "Disease Forecast": forecast,
        "Intervention Recommendations": intervention,
        "Risk Analysis": risk_analysis,
        "Prevention Plan": prevention,
    }
    return build_report_pdf("Complete Patient Clinical Summary", sections, patient_data)

# ─────────────────────────────────────────────
# SAMPLE DATA + ML
# ─────────────────────────────────────────────
def generate_sample_df():
    np.random.seed(42)
    n = 50

    diseases = ["Type 2 Diabetes","Hypertension","COPD","Heart Disease","CKD","Asthma"]

    # ✅ REAL INDIAN STATES
    regions = [
        "Andhra Pradesh", "Telangana", "Tamil Nadu",
        "Karnataka", "Maharashtra", "Delhi",
        "West Bengal", "Gujarat", "Rajasthan"
    ]

    df = pd.DataFrame({
        "Patient_ID":        [f"P{1000+i}" for i in range(n)],
        "Age":               np.random.randint(35, 80, n),
        "Gender":            np.random.choice(["Male","Female"], n),
        "BMI":               np.round(np.random.uniform(18, 42, n), 1),
        "Disease":           np.random.choice(diseases, n),
        "Stage":             np.random.choice(["Early","Moderate","Advanced"], n, p=[0.4,0.4,0.2]),
        "HbA1c":             np.round(np.random.uniform(4.5, 12, n), 1),
        "Systolic_BP":       np.random.randint(100, 180, n),
        "Cholesterol":       np.random.randint(140, 300, n),
        "eGFR":              np.random.randint(15, 120, n),
        "Comorbidities":     np.random.randint(0, 5, n),
        "Medications":       np.random.randint(1, 8, n),
        "Smoking":           np.random.choice(["Never","Former","Current"], n),
        "Physical_Activity": np.random.choice(["Sedentary","Moderate","Active"], n),

        # ✅ FIXED REGION
        "Region":            np.random.choice(regions, n),

        "Diagnosis_Year":    np.random.randint(2015, 2024, n),
        "Hospitalizations":  np.random.randint(0, 6, n),
        "Risk_Score":        np.round(np.random.uniform(10, 95, n), 1),
    })

    return df

def train_ml_model(df):
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    def risk_category(score):
        if score > 70:    return 2
        elif score >= 40: return 1
        else:             return 0
    df["Risk_Level"] = df["Risk_Score"].apply(risk_category)
    X = df.drop(["Risk_Score","Risk_Level"], axis=1, errors="ignore")
    y = df["Risk_Level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X.columns, label_encoders

# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
LIGHT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font_color="#1f2937",
    xaxis=dict(gridcolor="#e5e7eb"),
    yaxis=dict(gridcolor="#e5e7eb"),
)
def apply_light(fig):
    fig.update_layout(**LIGHT)
    return fig

def gauge_chart(value, title, max_val=100, color="#4f46e5"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#1e293b"}},
        number={"font": {"color": "#1e293b", "size": 26}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#64748b"},
            "bar":  {"color": color},
            "bgcolor": "#f1f5f9",
            "steps": [
                {"range": [0, max_val*0.33], "color": "#dcfce7"},
                {"range": [max_val*0.33, max_val*0.66], "color": "#fef9c3"},
                {"range": [max_val*0.66, max_val],       "color": "#fee2e2"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="#ffffff", font_color="#1e293b",
                      margin=dict(t=60,b=10,l=10,r=10), height=220)
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏥 Chronic Disease Forecasting Platform</h1>
  <p>AI-powered disease progression prediction · Timeline forecasting · Intervention recommendations</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📂 Data Input",
    "🔮 Disease Forecast",
    "💊 Interventions",
    "📊 Analytics Dashboard",
    "⚠️ Risk Stratification",
    "🗺️ Geospatial View",
    "👤 Patient Summary",
])

# ══════════════════════════════════════════════
# TAB 1 – DATA INPUT
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 📂 Patient Data Input")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Upload Excel / CSV File")
        uploaded_excel = st.file_uploader("Upload patient data (.xlsx / .csv)",
                                          type=["xlsx","xls","csv"], key="excel_up")
        if uploaded_excel:
            try:
                df = pd.read_csv(uploaded_excel) if uploaded_excel.name.endswith(".csv") \
                     else pd.read_excel(uploaded_excel)
                st.session_state.patient_df = df
                st.success(f"✅ Loaded {len(df)} patient records, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        if st.button("📥 Load Sample Dataset", use_container_width=True):
            st.session_state.patient_df = generate_sample_df()
            st.success("✅ Sample dataset with 50 patients loaded!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Upload Patient Report Image")
        uploaded_img = st.file_uploader("Upload report/lab image (.png/.jpg/.jpeg)",
                                        type=["png","jpg","jpeg"], key="img_up")
        if uploaded_img:
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded Report", use_container_width=True)
            st.session_state.image_data = img
            try:
                import pytesseract
                extracted = pytesseract.image_to_string(img)
                if extracted.strip():
                    st.markdown("**Extracted Text:**")
                    st.text_area("OCR Output", extracted, height=150)
                else:
                    st.info("No text extracted. Describe key values below.")
            except Exception:
                st.info("💡 pytesseract not installed — describe key values below.")
            img_notes = st.text_area("📝 Summarize key lab values from image", height=80,
                                     placeholder="e.g. HbA1c 8.2, BP 145/90, Creatinine 1.8...")
            if img_notes:
                st.session_state["img_notes"] = img_notes
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.patient_df is not None:
        st.markdown("---")
        st.markdown("#### 📋 Patient Data Preview")
        df = st.session_state.patient_df
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Patients", len(df))
        col_b.metric("Features", len(df.columns))
        if "Age" in df.columns:
            col_c.metric("Avg Age", f"{df['Age'].mean():.1f}")
        if "Risk_Score" in df.columns:
            col_d.metric("High Risk (>70)", int((df["Risk_Score"]>70).sum()))
        st.dataframe(df, use_container_width=True, height=300)

        st.markdown("#### 🔍 Data Quality Report")
        q1, q2 = st.columns(2)
        with q1:
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing):
                st.warning(f"⚠️ Missing values in {len(missing)} columns")
                st.dataframe(missing.rename("Missing Count"))
            else:
                st.success("✅ No missing values detected")
        with q2:
            st.markdown("**Numeric Summary**")
            st.dataframe(df.describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – DISEASE FORECAST
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🔮 Disease Progression Forecast")
    if st.session_state.patient_df is None:
        no_data_placeholder()
    else:
        df = st.session_state.patient_df
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("**Select Patient or Group**")
            mode = st.radio("Forecast mode", ["Individual Patient","Population Group"])
            if mode == "Individual Patient" and "Patient_ID" in df.columns:
                pid = st.selectbox("Patient ID", df["Patient_ID"].tolist())
                patient_row = df[df["Patient_ID"] == pid].iloc[0]
                context = patient_row.to_dict()
            else:
                disease_filter = "All"
                if "Disease" in df.columns:
                    disease_filter = st.selectbox("Disease Group", ["All"] + df["Disease"].unique().tolist())
                sub = df if disease_filter == "All" else df[df["Disease"] == disease_filter]
                context = {
                    "total_patients": len(sub),
                    "avg_age": round(sub["Age"].mean(), 1) if "Age" in sub.columns else "N/A",
                    "avg_risk_score": round(sub["Risk_Score"].mean(), 1) if "Risk_Score" in sub.columns else "N/A",
                    "disease_distribution": sub["Disease"].value_counts().to_dict() if "Disease" in sub.columns else {},
                    "stage_distribution":   sub["Stage"].value_counts().to_dict()   if "Stage"   in sub.columns else {},
                }
            horizon = st.selectbox("Forecast Horizon", ["6 months","1 year","2 years","5 years"])
            img_supplement = st.session_state.get("img_notes","")
            st.markdown('</div>', unsafe_allow_html=True)

        with col1:
            if st.button("🔮 Generate Forecast", use_container_width=True):
                prompt = f"""
Analyze the following patient/population data and provide a comprehensive chronic disease progression forecast.

PATIENT/POPULATION DATA:
{json.dumps(context, indent=2, default=str)}

IMAGE REPORT NOTES (if any): {img_supplement}
FORECAST HORIZON: {horizon}

Please provide:
1. **Current Status Assessment** — key risk factors, severity classification
2. **Disease Progression Forecast** — expected trajectory over {horizon} with confidence levels
3. **Milestone Timeline** — specific clinical milestones expected
4. **Biomarker Trends** — predicted changes in key biomarkers
5. **Complication Risk** — probability estimates for major complications
6. **Modifying Factors** — what could accelerate or decelerate progression
7. Add a clinical disclaimer.
"""
                with st.spinner("Analyzing with Llama 3.3..."):
                    result = call_groq(prompt)
                    st.session_state.forecast_result = result

        if st.session_state.forecast_result:
            st.markdown("---")
            st.markdown("#### 📋 Forecast Report")
            st.markdown(st.session_state.forecast_result)

            pdf_bytes = build_report_pdf(
                "Disease Progression Forecast Report",
                {"Forecast Analysis": st.session_state.forecast_result},
                patient_data=context if mode == "Individual Patient" else None
            )
            st.download_button(
                "📥 Download Forecast Report (PDF)",
                data=pdf_bytes,
                file_name="disease_forecast.pdf",
                mime="application/pdf"
            )

        st.markdown("---")
        st.markdown("#### 📈 Disease Progression Timeline (Simulated)")
        if "Risk_Score" in df.columns:
            months = list(range(0, 61, 6))
            base   = df["Risk_Score"].mean() if mode != "Individual Patient" else float(patient_row.get("Risk_Score", 50))
            no_intervention   = [min(100, base + i*0.8 + np.random.normal(0,1))   for i in range(len(months))]
            with_intervention = [min(100, base + i*0.2 + np.random.normal(0,0.5)) for i in range(len(months))]
            optimal           = [max(0,   base - i*0.5 + np.random.normal(0,0.3)) for i in range(len(months))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=no_intervention,   name="No Intervention",     line=dict(color="#ff4b4b",width=2.5)))
            fig.add_trace(go.Scatter(x=months, y=with_intervention, name="Standard Care",       line=dict(color="#ffa500",width=2.5)))
            fig.add_trace(go.Scatter(x=months, y=optimal,           name="Optimal Intervention",line=dict(color="#00c864",width=2.5)))
            fig.update_layout(title="Risk Score Trajectory Over 5 Years",
                              xaxis_title="Months", yaxis_title="Risk Score",
                              height=360, **LIGHT)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 – INTERVENTIONS
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 💊 Intervention Recommendations")
    if st.session_state.patient_df is None:
        no_data_placeholder()
    else:
        df = st.session_state.patient_df
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            if "Patient_ID" in df.columns:
                pid2  = st.selectbox("Select Patient", df["Patient_ID"].tolist(), key="pid2")
                pdata = df[df["Patient_ID"] == pid2].iloc[0].to_dict()
            else:
                pdata = df.iloc[0].to_dict()
            categories = st.multiselect("Intervention Categories",
                ["Pharmacological","Lifestyle","Monitoring","Surgical","Nutritional","Psychological","Community Support"],
                default=["Pharmacological","Lifestyle","Monitoring"])
            urgency = st.select_slider("Urgency Level", ["Routine","Soon","Urgent","Emergency"], "Soon")
            st.markdown('</div>', unsafe_allow_html=True)

        with col1:
            if st.button("💊 Generate Recommendations", use_container_width=True):
                prompt = f"""
Based on the patient profile below, provide evidence-based intervention recommendations.

PATIENT DATA: {json.dumps(pdata, indent=2, default=str)}
REQUESTED CATEGORIES: {', '.join(categories)}
URGENCY: {urgency}

Provide:
1. **Priority Interventions** (High/Medium/Low priority)
2. **Pharmacological Options** — drug classes, target doses, monitoring
3. **Lifestyle Modifications** — specific measurable goals
4. **Monitoring Schedule** — lab tests and frequency
5. **Referrals** — specialists recommended
6. **6-Month Treatment Goals** — measurable endpoints
Include priority levels (High/Medium/Low) and evidence grades (A/B/C).
"""
                with st.spinner("Generating clinical recommendations..."):
                    result = call_groq(prompt)
                    st.session_state.intervention_result = result

        if st.session_state.intervention_result:
            st.markdown("---")
            st.markdown("#### 📋 Intervention Plan")
            st.markdown(st.session_state.intervention_result)

            pdf_bytes = build_report_pdf(
                "Intervention Recommendations Report",
                {"Intervention Plan": st.session_state.intervention_result},
                patient_data=pdata
            )
            # st.download_button(
            #     "📥 Download Intervention Plan (PDF)",
            #     data=pdf_bytes,
            #     file_name="intervention_plan.pdf",
            #     mime="application/pdf"
            # )

        st.markdown("---")
        st.markdown("#### 📊 Projected Impact of Interventions")
        interventions_list = ["Medication Adherence","Diet Change","Exercise","Smoking Cessation",
                              "Blood Pressure Control","Glucose Management","Stress Reduction"]
        impact = [np.random.uniform(5, 35) for _ in interventions_list]
        fig = go.Figure(go.Bar(
            x=impact, y=interventions_list, orientation="h",
            marker=dict(color=px.colors.sequential.Plasma[::-1][:len(interventions_list)]),
            text=[f"{v:.1f}% risk ↓" for v in impact], textposition="outside"
        ))
        fig.update_layout(title="Estimated Risk Reduction per Intervention",
                          xaxis_title="% Risk Reduction", height=360, **LIGHT)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 – ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 📊 Analytics Dashboard")
    if st.session_state.patient_df is None:
        no_data_placeholder()
    else:
        df = st.session_state.patient_df

        st.markdown("#### 🎯 Key Performance Indicators")
        g1, g2, g3, g4 = st.columns(4)
        avg_risk      = df["Risk_Score"].mean()          if "Risk_Score"       in df.columns else 52.3
        high_risk_pct = (df["Risk_Score"]>70).mean()*100 if "Risk_Score"       in df.columns else 24.0
        avg_bmi       = df["BMI"].mean()                 if "BMI"              in df.columns else 28.4
        hosp_rate     = df["Hospitalizations"].mean()    if "Hospitalizations" in df.columns else 1.8

        with g1: st.plotly_chart(gauge_chart(round(avg_risk,1),      "Avg Risk Score",       100, "#ff4b4b"), use_container_width=True)
        with g2: st.plotly_chart(gauge_chart(round(high_risk_pct,1), "High-Risk %",          100, "#ffa500"), use_container_width=True)
        with g3: st.plotly_chart(gauge_chart(round(avg_bmi,1),       "Avg BMI",               50, "#4f46e5"), use_container_width=True)
        with g4: st.plotly_chart(gauge_chart(round(hosp_rate,1),     "Avg Hospitalizations",  10, "#6a11cb"), use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if "Disease" in df.columns:
                counts = df["Disease"].value_counts().reset_index()
                counts.columns = ["Disease","Count"]
                fig = px.pie(counts, names="Disease", values="Count", title="Disease Distribution",
                             color_discrete_sequence=px.colors.sequential.Plasma)
                apply_light(fig)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Stage" in df.columns and "Disease" in df.columns:
                stage_counts = df.groupby(["Disease","Stage"]).size().reset_index(name="Count")
                fig = px.bar(stage_counts, x="Disease", y="Count", color="Stage",
                             title="Disease Severity by Stage",
                             color_discrete_map={"Early":"#00c864","Moderate":"#ffa500","Advanced":"#ff4b4b"},
                             barmode="stack")
                apply_light(fig)
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Age" in df.columns and "Risk_Score" in df.columns:
                fig = px.scatter(df, x="Age", y="Risk_Score",
                                 color="Disease" if "Disease" in df.columns else None,
                                 size="BMI"      if "BMI"     in df.columns else None,
                                 title="Age vs Risk Score (bubble = BMI)",
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
                apply_light(fig)
                st.plotly_chart(fig, use_container_width=True)
        with c4:
            if "Diagnosis_Year" in df.columns:
                _cc = df.columns[0]
                trend = df.groupby("Diagnosis_Year").agg(
                    Patients=(_cc,"count"),
                    Avg_Risk=("Risk_Score","mean") if "Risk_Score" in df.columns else (_cc,"count")
                ).reset_index()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=trend["Diagnosis_Year"],y=trend["Patients"],
                                     name="New Cases",marker_color="rgba(101,31,255,0.6)"), secondary_y=False)
                fig.add_trace(go.Scatter(x=trend["Diagnosis_Year"],y=trend["Avg_Risk"],
                                         name="Avg Risk",line=dict(color="#ffa500",width=2.5)), secondary_y=True)
                fig.update_layout(title="Diagnosis Trends Over Years", **LIGHT, height=360)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔥 Biomarker Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 4:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            title="Feature Correlation Matrix")
            fig.update_layout(**LIGHT, height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📈 Biomarker Trend Simulation")
        months_sim = list(range(0, 25, 3))
        bio_fig = go.Figure()
        biomarkers  = {"HbA1c":7.5,"Systolic_BP":140,"Cholesterol":220,"eGFR":65}
        colors_bio  = ["#ff4b4b","#ffa500","#4f46e5","#00c864"]
        for (bio, start), col in zip(biomarkers.items(), colors_bio):
            vals = [start + np.random.normal(-0.3*i, 0.5) for i in range(len(months_sim))]
            bio_fig.add_trace(go.Scatter(x=months_sim, y=vals, name=bio,
                                         line=dict(color=col, width=2), mode="lines+markers"))
        bio_fig.update_layout(title="Biomarker Trends with Standard Care (Simulated)",
                              xaxis_title="Month", height=360, **LIGHT)
        st.plotly_chart(bio_fig, use_container_width=True)

        # ── ML ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Machine Learning Risk Prediction")
        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ `scikit-learn` not installed. Run `pip install scikit-learn`.")
        else:
            col_train, col_info = st.columns([1, 2])
            with col_train:
                if st.button("🏋️ Train ML Model", use_container_width=True):
                    with st.spinner("Training Random Forest..."):
                        model, acc, features, encoders = train_ml_model(df)
                    st.success(f"✅ Accuracy: **{acc:.0%}**")
                    st.session_state["ml_model"]    = model
                    st.session_state["ml_features"] = list(features)
                    st.session_state["ml_encoders"] = encoders

                import pickle, os
                st.markdown("---")
                cs, cl = st.columns(2)
                with cs:
                    if st.button("💾 Save Model", use_container_width=True):
                        if "ml_model" in st.session_state:
                            with open("saved_model.pkl","wb") as f:
                                pickle.dump({"model":st.session_state["ml_model"],
                                             "features":st.session_state["ml_features"],
                                             "encoders":st.session_state["ml_encoders"]}, f)
                            st.success("Saved!")
                        else:
                            st.warning("Train a model first.")
                with cl:
                    if st.button("📂 Load Model", use_container_width=True):
                        if os.path.exists("saved_model.pkl"):
                            with open("saved_model.pkl","rb") as f:
                                _d = pickle.load(f)
                            st.session_state["ml_model"]    = _d["model"]
                            st.session_state["ml_features"] = _d["features"]
                            st.session_state["ml_encoders"] = _d["encoders"]
                            st.rerun()
                        else:
                            st.warning("No saved model found.")

            with col_info:
                if "ml_model" in st.session_state:
                    model    = st.session_state["ml_model"]
                    features = st.session_state["ml_features"]
                    imp_df   = pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}
                                            ).sort_values("Importance",ascending=False)
                    st.markdown("**📊 Feature Importance**")
                    st.bar_chart(imp_df.set_index("Feature"))

            if "ml_model" in st.session_state:
                st.markdown("---")
                st.markdown("#### 🩺 Patient Risk Assessment Form")
                encoders    = st.session_state["ml_encoders"]
                model_feats = st.session_state["ml_features"]
                patient_input = {}
                form_cols = st.columns(3)
                col_idx   = 0
                for feat in model_feats:
                    if feat not in df.columns: continue
                    col_data = df[feat]
                    with form_cols[col_idx % 3]:
                        if feat in encoders:
                            options = list(encoders[feat].classes_)
                            patient_input[feat] = st.selectbox(feat.replace("_"," "), options, key=f"inp_{feat}")
                        else:
                            mn  = float(col_data.min())
                            mx  = float(col_data.max())
                            med = float(col_data.median())
                            patient_input[feat] = st.number_input(
                                feat.replace("_"," "), min_value=mn, max_value=mx,
                                value=med, step=round((mx-mn)/100,2) or 0.1, key=f"inp_{feat}")
                    col_idx += 1

                if st.button("🔮 Predict Risk & Get Prevention Steps", use_container_width=True):
                    row = pd.DataFrame([patient_input])
                    for c2, le in encoders.items():
                        if c2 in row.columns:
                            row[c2] = le.transform(row[c2].astype(str))
                    row  = row[[c2 for c2 in model_feats if c2 in row.columns]]
                    pred = st.session_state["ml_model"].predict(row)[0]
                    prob = st.session_state["ml_model"].predict_proba(row)[0]
                    label_map   = {0:"Low", 1:"Medium", 2:"High"}
                    risk_colors = {0:"#dcfce7",1:"#fef3c7",2:"#fee2e2"}
                    st.session_state["ml_pred_html"] = f"""<div style='padding:16px;border-radius:12px;
                        background:{risk_colors[pred]};margin-bottom:12px;'>
                        <h3 style='margin:0'>Predicted Risk Level: {label_map[pred]}</h3>
                        <p style='margin:4px 0 0'>Confidence: Low {prob[0]:.0%} | Medium {prob[1]:.0%} | High {prob[2]:.0%}</p>
                        </div>"""
                    risk_name = label_map[pred]
                    prev_prompt = f"""
A patient has been assessed at **{risk_name} risk** for chronic disease progression.
Patient details: {json.dumps(patient_input, indent=2, default=str)}
Model confidence: Low {prob[0]:.0%} | Medium {prob[1]:.0%} | High {prob[2]:.0%}

Provide:
1. **Risk Explanation** — why this patient is at {risk_name} risk
2. **Top 5 Prevention Steps** — specific, actionable steps
3. **Lifestyle Changes** — diet, exercise, sleep, stress
4. **Monitoring Schedule** — tests and frequency
5. **Warning Signs** — red flags and when to seek emergency care
6. **6-Month Goal** — measurable health targets
Use plain language the patient can understand. Include a disclaimer.
"""
                    with st.spinner("Generating prevention plan..."):
                        st.session_state["prevention_plan"] = call_groq(prev_prompt)

                if st.session_state.get("ml_pred_html"):
                    st.markdown(st.session_state["ml_pred_html"], unsafe_allow_html=True)
                if st.session_state.get("prevention_plan"):
                    st.markdown("#### 💊 Personalized Prevention Plan")
                    st.markdown(st.session_state["prevention_plan"])
                    pdf_bytes = build_report_pdf(
                        "Personalized Prevention Plan",
                        {"Prevention Plan": st.session_state["prevention_plan"]}
                    )
                    # st.download_button(
                    #     "📥 Download Prevention Plan (PDF)",
                    #     data=pdf_bytes,
                    #     file_name="prevention_plan.pdf",
                    #     mime="application/pdf"
                    # )

# ══════════════════════════════════════════════
# TAB 5 – RISK STRATIFICATION
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown("### ⚠️ Risk Stratification & Population Health")
    if st.session_state.patient_df is None:
        no_data_placeholder()
    else:
        df = st.session_state.patient_df

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### 🏷️ Risk Tiers")
            if "Risk_Score" in df.columns:
                high_r = df[df["Risk_Score"] > 70]
                med_r  = df[(df["Risk_Score"] >= 40) & (df["Risk_Score"] <= 70)]
                low_r  = df[df["Risk_Score"] < 40]
                st.markdown(f'<div class="risk-high">🔴 <b>High Risk</b>: {len(high_r)} patients ({len(high_r)/len(df)*100:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="risk-medium">🟡 <b>Medium Risk</b>: {len(med_r)} patients ({len(med_r)/len(df)*100:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="risk-low">🟢 <b>Low Risk</b>: {len(low_r)} patients ({len(low_r)/len(df)*100:.1f}%)</div>', unsafe_allow_html=True)

            st.markdown("---")
            if st.button("🤖 AI Population Analysis", use_container_width=True):
                pop_context = {
                    "total": len(df),
                    "avg_risk": round(df["Risk_Score"].mean(),1) if "Risk_Score" in df.columns else "N/A",
                    "high_risk_n": len(df[df["Risk_Score"]>70]) if "Risk_Score" in df.columns else "N/A",
                    "disease_mix": df["Disease"].value_counts().to_dict() if "Disease" in df.columns else {},
                    "avg_age": round(df["Age"].mean(),1) if "Age" in df.columns else "N/A",
                }
                prompt = f"""
Perform a population health analysis for this chronic disease cohort:
{json.dumps(pop_context, indent=2, default=str)}

Provide:
1. **Population Risk Summary**
2. **High-Risk Patient Profile**
3. **Disease Hotspots**
4. **Resource Allocation Recommendations**
5. **Prevention Priorities**
6. **30-60-90 Day Action Plan**
7. **Expected Outcomes**
"""
                with st.spinner("Running population health analysis..."):
                    result = call_groq(prompt)
                    st.session_state.risk_result = result
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if "Risk_Score" in df.columns:
                fig = px.histogram(df, x="Risk_Score", nbins=20,
                                   color_discrete_sequence=["#4f46e5"],
                                   title="Risk Score Distribution")
                fig.add_vline(x=40, line_dash="dash", line_color="#ffa500", annotation_text="Medium threshold")
                fig.add_vline(x=70, line_dash="dash", line_color="#ff4b4b", annotation_text="High threshold")
                apply_light(fig)
                st.plotly_chart(fig, use_container_width=True)
            if "Disease" in df.columns and "Risk_Score" in df.columns:
                risk_by_disease = df.groupby("Disease")["Risk_Score"].mean().sort_values(ascending=False).reset_index()
                fig2 = px.bar(risk_by_disease, x="Risk_Score", y="Disease", orientation="h",
                              color="Risk_Score", color_continuous_scale="Reds",
                              title="Average Risk Score by Disease")
                apply_light(fig2)
                st.plotly_chart(fig2, use_container_width=True)

        if st.session_state.risk_result:
            st.markdown("---")
            st.markdown("#### 🤖 AI Population Health Analysis")
            st.markdown(st.session_state.risk_result)
            pdf_bytes = build_report_pdf(
                "Population Health Analysis Report",
                {"Population Analysis": st.session_state.risk_result}
            )
            # st.download_button(
            #     "📥 Download Population Analysis (PDF)",
            #     data=pdf_bytes,
            #     file_name="population_analysis.pdf",
            #     mime="application/pdf"
            # )

        st.markdown("---")
        st.markdown("#### 🚨 High-Risk Patient List")
        if "Risk_Score" in df.columns:
            high_df = df[df["Risk_Score"] > 70].sort_values("Risk_Score", ascending=False)
            if len(high_df):
                st.dataframe(high_df, use_container_width=True, height=250)
            else:
                st.info("No high-risk patients in current dataset.")

# ══════════════════════════════════════════════
# TAB 6 – GEOSPATIAL
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🗺️ Geospatial Disease Distribution")
    if st.session_state.patient_df is None:
        no_data_placeholder()
    else:
        df = st.session_state.patient_df
        st.info("💡 Interactive bubble map of disease burden by region. Upload a GeoJSON below for full choropleth mapping.")

        if "Region" in df.columns:
            _rc = df.columns[0]
            _agg = {"Patients": (_rc,"count")}
            if "Risk_Score" in df.columns:
                _agg["Avg_Risk"]  = ("Risk_Score","mean")
                _agg["High_Risk"] = ("Risk_Score", lambda x: (x>70).sum())
            region_summary = df.groupby("Region").agg(**_agg).reset_index().round(1)
            if "Avg_Risk" not in region_summary.columns:
                region_summary["Avg_Risk"]  = 0
                region_summary["High_Risk"] = 0

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(region_summary, x="Region", y="Patients",
                             color="Avg_Risk", color_continuous_scale="Reds",
                             title="Patients per Region")
                apply_light(fig)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.pie(region_summary, names="Region", values="High_Risk",
                              title="High-Risk Patients by Region",
                              color_discrete_sequence=px.colors.sequential.Plasma)
                apply_light(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### 📊 Regional Breakdown Table")
            st.dataframe(region_summary, use_container_width=True)

            region_coords = {
                "North":   (28.7041, 77.1025), "South":   (13.0827, 80.2707),
                "East":    (22.5726, 88.3639), "West":    (19.0760, 72.8777),
                "Central": (21.2514, 81.6296),
            }
            region_summary["lat"] = region_summary["Region"].map(lambda r: region_coords.get(r,(20,78))[0])
            region_summary["lon"] = region_summary["Region"].map(lambda r: region_coords.get(r,(20,78))[1])
            fig_map = px.scatter_mapbox(
                region_summary, lat="lat", lon="lon",
                size="Patients", color="Avg_Risk",
                color_continuous_scale="Reds",
                hover_name="Region",
                hover_data={"Patients":True,"Avg_Risk":True,"High_Risk":True},
                zoom=4, center={"lat":20.5937,"lon":78.9629},
                mapbox_style="carto-darkmatter",
                title="Regional Disease Burden Map", size_max=50,
            )
            fig_map.update_layout(paper_bgcolor="white", font_color="#1f2937",
                                  height=500, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### 🗺️ Upload Custom GeoJSON")
        shp_file = st.file_uploader("Upload GeoJSON for custom choropleth map",
                                    type=["geojson","json"], key="geo_up")
        if shp_file:
            try:
                import geopandas as gpd
                geojson_data = json.load(shp_file)

# Create dummy region_summary if not already
                if "Region" in df.columns:
                    region_summary = df.groupby("Region").agg(
                        Patients=("Patient_ID","count"),
                        Avg_Risk=("Risk_Score","mean")
                    ).reset_index()
                else:
                    st.warning("Region column missing in dataset")
                    st.stop()

                fig = px.choropleth_mapbox(
                    region_summary,
                    geojson=geojson_data,
                    locations="Region",
                    featureidkey="properties.state",
                    color="Avg_Risk",
                    color_continuous_scale="Reds",
                    mapbox_style="carto-positron",  # lighter theme
                    zoom=4,
                    center={"lat": 20.5937, "lon": 78.9629},
                    opacity=0.6,
                )

                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading GeoFile: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 7 – PATIENT SUMMARY  (new)
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 👤 Complete Patient Summary")
    if st.session_state.patient_df is None:
        no_data_placeholder()
    else:
        df = st.session_state.patient_df

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Select a Patient")
        if "Patient_ID" in df.columns:
            selected_pid = st.selectbox("Patient ID", df["Patient_ID"].tolist(), key="summary_pid")
            patient_row  = df[df["Patient_ID"] == selected_pid].iloc[0]
        else:
            patient_row = df.iloc[0]
        patient_dict = patient_row.to_dict()

        # ── Profile cards ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🧾 Patient Profile")
        num_fields = len(patient_dict)
        cols = st.columns(4)
        for idx, (k, v) in enumerate(patient_dict.items()):
            with cols[idx % 4]:
                st.metric(k.replace("_"," "), str(v))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Risk gauge ───────────────────────────────────────────────────
        if "Risk_Score" in patient_dict:
            risk_val = float(patient_dict["Risk_Score"])
            risk_col = "#ef4444" if risk_val > 70 else ("#f59e0b" if risk_val >= 40 else "#22c55e")
            st.markdown("---")
            c_g, c_r = st.columns([1, 2])
            with c_g:
                st.plotly_chart(gauge_chart(risk_val, "Patient Risk Score", 100, risk_col), use_container_width=True)
            with c_r:
                risk_label = "🔴 HIGH RISK" if risk_val > 70 else ("🟡 MEDIUM RISK" if risk_val >= 40 else "🟢 LOW RISK")
                risk_bg    = "#fee2e2" if risk_val > 70 else ("#fef3c7" if risk_val >= 40 else "#dcfce7")
                st.markdown(f"""
                <div style='background:{risk_bg};border-radius:14px;padding:24px;margin-top:10px;'>
                  <h2 style='margin:0'>{risk_label}</h2>
                  <p style='margin:8px 0 0;color:#374151'>Risk Score: <b>{risk_val}</b> / 100</p>
                  {"<p style='color:#374151'>This patient requires <b>immediate clinical attention</b>.</p>" if risk_val > 70 else ""}
                </div>
                """, unsafe_allow_html=True)

        # ── Generate all AI sections ──────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🤖 Generate Full AI Clinical Report")
        horizon_s = st.selectbox("Forecast horizon for summary", ["6 months","1 year","2 years","5 years"], key="sum_hor")

        if st.button("🚀 Generate Complete Patient Report", use_container_width=True):
            with st.spinner("Generating Forecast..."):
                forecast_prompt = f"""
Provide a comprehensive disease progression forecast for:
PATIENT DATA: {json.dumps(patient_dict, indent=2, default=str)}
HORIZON: {horizon_s}
Include: Current Status, Progression Forecast, Milestone Timeline, Biomarker Trends,
Complication Risk, Modifying Factors. Add clinical disclaimer.
"""
                st.session_state["sum_forecast"] = call_groq(forecast_prompt)

            with st.spinner("Generating Intervention Plan..."):
                intv_prompt = f"""
Provide evidence-based intervention recommendations for:
PATIENT DATA: {json.dumps(patient_dict, indent=2, default=str)}
Include: Priority Interventions, Pharmacological Options, Lifestyle Modifications,
Monitoring Schedule, Referrals, 6-Month Goals. Include evidence grades.
"""
                st.session_state["sum_intervention"] = call_groq(intv_prompt)

            with st.spinner("Generating Risk Analysis..."):
                risk_prompt = f"""
Perform an individual patient risk analysis for:
PATIENT DATA: {json.dumps(patient_dict, indent=2, default=str)}
Include: Risk Stratification, Contributing Factors, Short-term vs Long-term Risks,
Protective Factors, Recommended Screening. Add clinical disclaimer.
"""
                st.session_state["sum_risk"] = call_groq(risk_prompt)

            with st.spinner("Generating Prevention Plan..."):
                prev_prompt = f"""
Create a personalized prevention and management plan for:
PATIENT DATA: {json.dumps(patient_dict, indent=2, default=str)}
Include: Top 5 Prevention Steps, Lifestyle Changes, Monitoring Schedule,
Warning Signs, 6-Month Goals. Use patient-friendly language. Add disclaimer.
"""
                st.session_state["sum_prevention"] = call_groq(prev_prompt)

            st.success("✅ Complete report generated!")

        # ── Display sections ──────────────────────────────────────────────
        for key, label, icon in [
            ("sum_forecast",     "Disease Progression Forecast",    "🔮"),
            ("sum_intervention", "Intervention Recommendations",    "💊"),
            ("sum_risk",         "Risk Analysis",                   "⚠️"),
            ("sum_prevention",   "Personalized Prevention Plan",    "🛡️"),
        ]:
            if st.session_state.get(key):
                with st.expander(f"{icon} {label}", expanded=True):
                    st.markdown(st.session_state[key])

        # ── Single PDF download ───────────────────────────────────────────
        if all(st.session_state.get(k) for k in ["sum_forecast","sum_intervention","sum_risk","sum_prevention"]):
            st.markdown("---")
            st.markdown("#### 📄 Download Complete Patient Report")

            pdf_bytes = build_patient_summary_pdf(
                patient_data    = patient_dict,
                forecast        = st.session_state["sum_forecast"],
                intervention    = st.session_state["sum_intervention"],
                risk_analysis   = st.session_state["sum_risk"],
                prevention      = st.session_state["sum_prevention"],
            )

            pid_label = patient_dict.get("Patient_ID", "patient")
            st.download_button(
                label    = f"📥 Download Complete PDF Report for {pid_label}",
                data     = pdf_bytes,
                file_name= f"complete_report_{pid_label}.pdf",
                mime     = "application/pdf",
                use_container_width=True,
            )
            st.markdown("""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px;margin-top:8px;'>
              ✅ The PDF includes: patient profile table, disease forecast, intervention plan,
              risk analysis, and prevention plan — all in one document.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.8rem; padding:16px;">
  🏥 Chronic Disease Forecasting Platform &nbsp;|&nbsp; Powered by Groq · Llama 3.3 · Streamlit
  <br>⚠️ For clinical decision support only. All AI predictions require validation by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)