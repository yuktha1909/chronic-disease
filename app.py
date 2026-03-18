###############################################################################
# Diabetes Risk Prediction – Full Streamlit App
# Features: Risk Score, SHAP, PDF Report, Health Recommendations,
#           Multi-Model Comparison, BMI Calculator, Data Explorer
###############################################################################

import pathlib, warnings, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.stApp { background: #f7f4f0; }

.risk-card {
    background: white; border-radius: 16px; padding: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08); text-align: center; margin: 1rem 0;
}
.risk-low    { border-top: 6px solid #22c55e; }
.risk-moderate { border-top: 6px solid #f59e0b; }
.risk-high   { border-top: 6px solid #ef4444; }

.rec-card {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    padding: 0.8rem 1.2rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; font-size: 0.9rem;
}
.rec-card.warn   { background: #fffbeb; border-left-color: #f59e0b; }
.rec-card.danger { background: #fef2f2; border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Data & Model (Cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        return pd.read_csv(url)

@st.cache_resource
def train_models(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    feature_names = X.columns.tolist()
    preprocess = ColumnTransformer(
        [("num", StandardScaler(), feature_names)], remainder="drop"
    )
    cfgs = {
        "Logistic Regression": {
            "key": "log_reg",
            "est": LogisticRegression(max_iter=200),
            "grid": {"log_reg__C": [0.1, 1, 10], "log_reg__penalty": ["l2"]}
        },
        "Random Forest": {
            "key": "rf",
            "est": RandomForestClassifier(random_state=RANDOM_STATE),
            "grid": {"rf__n_estimators": [100, 300], "rf__max_depth": [None, 5, 10]}
        }
    }
    results = {}
    for name, cfg in cfgs.items():
        k = cfg["key"]
        pipe = Pipeline([("pre", preprocess), (k, cfg["est"])])
        gs = GridSearchCV(pipe, cfg["grid"], cv=5, scoring="f1", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        results[name] = {
            "model": best, "key": k, "y_pred": y_pred,
            "metrics": {
                "Accuracy":  accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall":    recall_score(y_test, y_pred),
                "F1 Score":  f1_score(y_test, y_pred)
            }
        }
    champion = max(results, key=lambda k: results[k]["metrics"]["F1 Score"])
    return results, champion, X_test, y_test, feature_names

# ──────────────────────────────────────────────
# Recommendations
# ──────────────────────────────────────────────
def get_recommendations(inputs, risk_pct):
    recs = []
    if inputs["Glucose"] > 125:
        recs.append({"text": "⚠️ High glucose – reduce sugar/refined carbs, monitor blood sugar.", "level": "danger"})
    elif inputs["Glucose"] > 100:
        recs.append({"text": "🟡 Borderline glucose – limit sugary drinks and processed foods.", "level": "warn"})
    else:
        recs.append({"text": "✅ Glucose is healthy. Maintain a balanced diet.", "level": "ok"})

    if inputs["BMI"] > 30:
        recs.append({"text": "⚠️ BMI indicates obesity – aim for 150 min/week exercise, consult a dietitian.", "level": "danger"})
    elif inputs["BMI"] > 25:
        recs.append({"text": "🟡 Slightly high BMI – regular walks and portion control can help.", "level": "warn"})
    else:
        recs.append({"text": "✅ BMI is healthy. Keep up current habits.", "level": "ok"})

    if inputs["BloodPressure"] > 90:
        recs.append({"text": "⚠️ High blood pressure – reduce salt, avoid smoking, see a doctor.", "level": "danger"})
    elif inputs["BloodPressure"] > 80:
        recs.append({"text": "🟡 Slightly elevated BP – manage stress and stay hydrated.", "level": "warn"})

    if inputs["DiabetesPedigreeFunction"] > 0.5:
        recs.append({"text": "🧬 Family history factor detected – routine check-ups are advised.", "level": "warn"})

    if inputs["Age"] > 45 and risk_pct > 50:
        recs.append({"text": "🔔 Age 45+ with elevated risk – annual HbA1c screening is recommended.", "level": "warn"})

    if risk_pct >= 70:
        recs.append({"text": "🚨 High risk – please consult an endocrinologist as soon as possible.", "level": "danger"})
    elif risk_pct >= 40:
        recs.append({"text": "📋 Moderate risk – lifestyle changes now can prevent Type 2 diabetes.", "level": "warn"})
    else:
        recs.append({"text": "💚 Low overall risk – maintain healthy habits and annual check-ups.", "level": "ok"})

    return recs

# ──────────────────────────────────────────────
# PDF Generator
# ──────────────────────────────────────────────
def generate_pdf(name, inputs, prediction, risk_pct, risk_level, recs):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.8*inch, rightMargin=0.8*inch,
                            topMargin=1*inch, bottomMargin=0.8*inch)
    styles = getSampleStyleSheet()
    story = []

    title_s = ParagraphStyle("t", parent=styles["Title"], fontSize=22,
                              textColor=colors.HexColor("#1e1e2e"), spaceAfter=4)
    sub_s   = ParagraphStyle("s", parent=styles["Normal"], fontSize=11,
                              textColor=colors.grey, spaceAfter=20)
    h2_s    = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                              textColor=colors.HexColor("#1e1e2e"), spaceBefore=14)
    body_s  = ParagraphStyle("b", parent=styles["Normal"], fontSize=10, leading=16)

    story += [
        Paragraph("Diabetes Risk Assessment Report", title_s),
        Paragraph(f"Prepared for: <b>{name}</b>", sub_s),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")),
        Spacer(1, 0.2*inch)
    ]

    cmap = {"Low": colors.HexColor("#22c55e"),
            "Moderate": colors.HexColor("#f59e0b"),
            "High": colors.HexColor("#ef4444")}
    t = Table([["Prediction","Risk Score","Risk Level"],
               [prediction, f"{risk_pct:.1f}%", risk_level]],
              colWidths=[2*inch, 2*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e1e2e")),
        ("TEXTCOLOR", (0,0),(-1,0),colors.white),
        ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1),11),
        ("ALIGN",     (0,0),(-1,-1),"CENTER"),
        ("VALIGN",    (0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f9fafb")]),
        ("TEXTCOLOR", (2,1),(2,1), cmap.get(risk_level, colors.grey)),
        ("FONTNAME",  (0,1),(-1,-1),"Helvetica-Bold"),
        ("FONTSIZE",  (0,1),(-1,-1),13),
        ("ROWHEIGHT", (0,1),(-1,-1),30),
        ("GRID",      (0,0),(-1,-1),0.5,colors.HexColor("#e5e7eb")),
    ]))
    story += [t, Spacer(1, 0.2*inch)]

    story.append(Paragraph("Patient Input Values", h2_s))
    t2 = Table([["Parameter","Value"]] + [[k, str(v)] for k, v in inputs.items()],
               colWidths=[3*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND",     (0,0),(-1,0),colors.HexColor("#f3f4f6")),
        ("FONTNAME",       (0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",       (0,0),(-1,-1),10),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#f9fafb")]),
        ("GRID",           (0,0),(-1,-1),0.5,colors.HexColor("#e5e7eb")),
        ("ALIGN",          (1,0),(1,-1),"CENTER"),
    ]))
    story += [t2, Spacer(1, 0.2*inch)]

    story.append(Paragraph("Health Recommendations", h2_s))
    for r in recs:
        clean = r["text"].lstrip("⚠️🟡✅🔔🧬🚨📋💚").strip()
        story += [Paragraph(f"• {clean}", body_s), Spacer(1, 4)]

    story += [
        Spacer(1, 0.3*inch),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")),
        Spacer(1, 0.1*inch),
        Paragraph("<i>Disclaimer: AI-generated report. Not a substitute for professional medical advice.</i>",
                  ParagraphStyle("d", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.grey, alignment=1))
    ]
    doc.build(story)
    return buf.getvalue()

# ──────────────────────────────────────────────
# Load & Train
# ──────────────────────────────────────────────
df = load_data()
with st.spinner("Training models… (cached after first run)"):
    results, champion_name, X_test, y_test, feature_names = train_models(df)

champion_model = results[champion_name]["model"]
champion_key   = results[champion_name]["key"]

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk")
    st.markdown("---")
    page = st.radio("Navigate", ["🔮 Predict", "📊 Model Comparison", "📈 Data Explorer"])
    st.markdown("---")
    st.markdown("**Champion Model**")
    st.success(champion_name)
    st.metric("F1 Score", f"{results[champion_name]['metrics']['F1 Score']:.3f}")

# ═══════════════════════════════════════════════
# PAGE 1 – PREDICT
# ═══════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Enter patient details to assess diabetes risk.")
    st.markdown("---")

    col_form, col_bmi = st.columns([2, 1])

    with col_bmi:
        st.markdown("### 🧮 BMI Calculator")
        h = st.number_input("Height (cm)", 100, 250, 165)
        w = st.number_input("Weight (kg)", 30, 200, 70)
        bmi_calc = w / ((h / 100) ** 2)
        cat = ("Underweight" if bmi_calc < 18.5 else "Normal" if bmi_calc < 25
               else "Overweight" if bmi_calc < 30 else "Obese")
        st.metric("Calculated BMI", f"{bmi_calc:.1f}", cat)
        st.info("Use this BMI value in the form →")

    with col_form:
        st.markdown("### 👤 Patient Information")
        patient_name = st.text_input("Patient Name", "Patient")
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose     = st.number_input("Glucose (mg/dL)", 0, 300, 110)
            bp          = st.number_input("Blood Pressure (mmHg)", 0, 150, 72)
            skin        = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with c2:
            insulin = st.number_input("Insulin (μU/mL)", 0, 900, 79)
            bmi     = st.number_input("BMI", 0.0, 70.0, float(f"{bmi_calc:.1f}"))
            dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47)
            age     = st.number_input("Age", 1, 120, 33)

    st.markdown("---")

    if st.button("🔮 Predict Diabetes Risk", use_container_width=True, type="primary"):
        inputs = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": bp, "SkinThickness": skin,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }
        input_df   = pd.DataFrame([inputs])
        prob       = champion_model.predict_proba(input_df)[0][1]
        risk_pct   = prob * 100
        pred_label = "Diabetic" if prob >= 0.5 else "Non-Diabetic"
        risk_level = "Low" if risk_pct < 35 else ("Moderate" if risk_pct < 65 else "High")
        emoji_map  = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}

        st.markdown(f"""
        <div class="risk-card risk-{risk_level.lower()}">
            <h1 style="font-size:3rem;margin:0">{emoji_map[risk_level]}</h1>
            <h2 style="margin:0.5rem 0">{pred_label}</h2>
            <p style="font-size:2.5rem;font-weight:700;margin:0">{risk_pct:.1f}%</p>
            <p style="color:gray">Risk Level: <strong>{risk_level}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        recs = get_recommendations(inputs, risk_pct)
        st.markdown("### 💡 Health Recommendations")
        for r in recs:
            css = "rec-card" + (" warn" if r["level"]=="warn" else " danger" if r["level"]=="danger" else "")
            st.markdown(f'<div class="{css}">{r["text"]}</div>', unsafe_allow_html=True)

        # SHAP
        st.markdown("### 🔍 Why this prediction? (SHAP Feature Importance)")
        try:
            pre   = champion_model.named_steps["pre"]
            model = champion_model.named_steps[champion_key]
            X_t   = pre.transform(input_df)
            if hasattr(X_t, "toarray"):
                X_t = X_t.toarray()
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_t)
            sv = np.array(shap_values)
            sv_plot = sv[:, :, 1] if sv.ndim == 3 else sv
            fig, _ = plt.subplots(figsize=(8, 4))
            shap.summary_plot(sv_plot, X_t, feature_names=feature_names,
                              plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

        # PDF
        st.markdown("### 📄 Download Report")
        pdf_bytes = generate_pdf(patient_name, inputs, pred_label, risk_pct, risk_level, recs)
        st.download_button(
            "⬇️ Download PDF Health Report", data=pdf_bytes,
            file_name=f"diabetes_report_{patient_name.replace(' ','_')}.pdf",
            mime="application/pdf", use_container_width=True
        )

# ═══════════════════════════════════════════════
# PAGE 2 – MODEL COMPARISON
# ═══════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("---")

    metric_df = pd.DataFrame({n: d["metrics"] for n, d in results.items()}).T.reset_index()
    metric_df.rename(columns={"index": "Model"}, inplace=True)
    st.dataframe(
        metric_df.style
            .format({c: "{:.3f}" for c in ["Accuracy","Precision","Recall","F1 Score"]})
            .highlight_max(axis=0, subset=["Accuracy","Precision","Recall","F1 Score"],
                           color="#d1fae5"),
        use_container_width=True, hide_index=True
    )

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x, width = np.arange(len(metrics)), 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f7f4f0"); ax.set_facecolor("#f7f4f0")
    for i, (name, d) in enumerate(results.items()):
        vals = [d["metrics"][m] for m in metrics]
        bars = ax.bar(x + i*width - width/2, vals, width, label=name,
                      color=["#6366f1","#22c55e"][i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score"); ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### Confusion Matrices")
    cols = st.columns(len(results))
    for col, (name, d) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{name}**")
            cm = confusion_matrix(y_test, d["y_pred"])
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            fig2.patch.set_facecolor("#f7f4f0")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
                        xticklabels=["No","Diabetes"], yticklabels=["No","Diabetes"])
            ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

# ═══════════════════════════════════════════════
# PAGE 3 – DATA EXPLORER
# ═══════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Diabetic", int(df["Outcome"].sum()))
    c3.metric("Non-Diabetic", int((df["Outcome"]==0).sum()))
    c4.metric("Prevalence", f"{df['Outcome'].mean()*100:.1f}%")

    st.markdown("### Feature Distribution by Outcome")
    feat = st.selectbox("Select Feature", feature_names)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f7f4f0"); ax.set_facecolor("#f7f4f0")
    for outcome, color, label in [(0,"#6366f1","No Diabetes"),(1,"#ef4444","Diabetes")]:
        ax.hist(df[df["Outcome"]==outcome][feat], bins=25, alpha=0.6,
                color=color, label=label, edgecolor="white")
    ax.set_xlabel(feat); ax.set_ylabel("Count")
    ax.set_title(f"{feat} Distribution"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    fig3.patch.set_facecolor("#f7f4f0")
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax3, linewidths=0.5)
    ax3.set_title("Feature Correlation Matrix", pad=15)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.markdown("### Raw Dataset")
    st.dataframe(df, use_container_width=True)