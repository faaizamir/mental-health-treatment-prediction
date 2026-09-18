"""
Mental Health Treatment Prediction — Streamlit app.

Loads the pre-trained model + scaler + encoders (produced by
train_and_save_model.py) and serves interactive predictions with a
SHAP-based explanation of each individual prediction.

Run locally:
    streamlit run app.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ARTIFACT_DIR = "artifacts"

st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/model.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
    encoders = joblib.load(f"{ARTIFACT_DIR}/encoders.joblib")
    with open(f"{ARTIFACT_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{ARTIFACT_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{ARTIFACT_DIR}/baseline_comparison.json") as f:
        baseline_comparison = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, scaler, encoders, feature_columns, metrics, baseline_comparison, explainer


model, scaler, encoders, feature_columns, metrics, baseline_comparison, explainer = load_artifacts()

# Human-friendly labels for the form (raw column name -> display label)
LABELS = {
    "Gender": "Gender",
    "Country": "Country",
    "Occupation": "Occupation",
    "self_employed": "Are you self-employed?",
    "family_history": "Family history of mental illness?",
    "Days_Indoors": "How many days have you stayed indoors recently?",
    "Growing_Stress": "Is your stress level growing?",
    "Changes_Habits": "Have your habits changed recently?",
    "Mental_Health_History": "Personal history of mental health issues?",
    "Mood_Swings": "How would you describe your mood swings?",
    "Coping_Struggles": "Are you struggling to cope?",
    "Work_Interest": "Have you lost interest in work?",
    "Social_Weakness": "Do you feel social weakness / withdrawal?",
    "mental_health_interview": "Comfortable discussing mental health in a job interview?",
    "care_options": "Aware of your employer's mental health care options?",
}

st.title("🧠 Mental Health Treatment Predictor")
st.caption(
    "Estimates the likelihood that someone will seek mental health treatment, "
    "based on a tuned XGBoost model (ROC-AUC ≈ "
    f"{metrics['roc_auc']:.2f}) trained on a public mental health survey dataset."
)

with st.sidebar:
    st.header("📊 Model Performance")
    st.caption(f"Final model: **{metrics['model_name']}** (tuned)")
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("Precision", f"{metrics['precision']:.1%}")
    st.metric("Recall", f"{metrics['recall']:.1%}")
    st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.divider()
    st.caption(
        "Metrics are computed on a held-out 20% test set the model never "
        "saw during training or tuning."
    )

    with st.expander(f"Why {metrics['model_name']}, not the other two?"):
        st.caption(
            "Three different algorithms were trained on identical data and "
            "compared on the same metrics — the table below is that "
            "comparison (before tuning). The one with the best ROC-AUC "
            "(the metric least sensitive to an arbitrary 0.5 cutoff) was "
            "carried forward and tuned further; the numbers above already "
            f"reflect that tuning, which improved {metrics['model_name']} "
            "further from its baseline shown here."
        )
        comp_df = pd.DataFrame(baseline_comparison).set_index("model_name")
        comp_df = comp_df.sort_values("roc_auc", ascending=False)
        st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)

with st.form("prediction_form"):
    st.subheader("Tell us about yourself")
    user_input = {}
    for col in feature_columns:
        options = list(encoders[col].classes_)
        user_input[col] = st.selectbox(LABELS.get(col, col), options, key=col)
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row DataFrame in the exact training feature order
    row = {col: encoders[col].transform([user_input[col]])[0] for col in feature_columns}
    X_input = pd.DataFrame([row], columns=feature_columns)
    X_input_scaled = pd.DataFrame(
        scaler.transform(X_input), columns=feature_columns
    )

    proba = float(model.predict_proba(X_input_scaled)[0, 1])
    prediction = "Likely to seek treatment" if proba >= 0.5 else "Unlikely to seek treatment"

    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Prediction", prediction)
    col2.metric("Probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    st.subheader("Why the model predicted this")
    st.caption(
        "Each bar shows how much a feature pushed this specific prediction "
        "toward (red) or away from (blue) 'will seek treatment'."
    )
    shap_values = explainer(X_input_scaled)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)

    with st.expander("How to read this chart"):
        st.markdown(
            "This is a **SHAP waterfall plot** — it explains *this one* "
            "prediction, not the model in general.\n\n"
            "- **`E[f(x)]` at the bottom** is the baseline: the average "
            "prediction across everyone in the dataset, before knowing "
            "anything about this specific person.\n"
            "- **Each bar is one feature**, listed from most to least "
            "influential for this prediction. The number next to each "
            "feature is the value *you* entered for it.\n"
            "- **Red bars push the prediction up** (toward *will* seek "
            "treatment); **blue bars push it down** (toward *won't*).\n"
            "- **The bars stack on top of each other**, starting from the "
            "baseline at the bottom and ending at **`f(x)`** at the top — "
            "that final number is this person's actual predicted "
            "probability.\n\n"
            "In short: it's a running total that shows exactly which "
            "answers moved the prediction, by how much, and in which "
            "direction — rather than just handing you a single probability "
            "with no explanation of where it came from."
        )

    st.info(
        "This is a demo built on a public survey dataset for a data analytics "
        "portfolio project — it is not a medical or diagnostic tool.",
        icon="ℹ️",
    )

st.divider()
st.caption(
    f"Model: tuned {metrics['model_name']} · "
    f"Accuracy: {metrics['accuracy']:.2f} · "
    f"Precision: {metrics['precision']:.2f} · "
    f"Recall: {metrics['recall']:.2f} · "
    f"F1: {metrics['f1_score']:.2f} · "
    f"ROC-AUC: {metrics['roc_auc']:.2f}"
)
