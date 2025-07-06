import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_BUNDLE_FILE_ID = "1W-69hWXBZomnUE6X37jOCKH8YuLAiA1h"

# Download model_bundle.pkl from Google Drive if not present
os.makedirs("models", exist_ok=True)
if not os.path.exists("models/model_bundle.pkl"):
    url = f"https://drive.google.com/uc?id={MODEL_BUNDLE_FILE_ID}"
    gdown.download(url, "models/model_bundle.pkl", quiet=False)

# Load all assets
bundle = joblib.load("models/model_bundle.pkl")
models = bundle["models"]
le_dict = bundle["le_dict"]
le_target = bundle["le_target"]
scaler = bundle["scaler"]
feature_labels = bundle["feature_labels"]
feature_stats = bundle["feature_stats"]
cat_cols = bundle["cat_cols"]
num_cols = bundle["num_cols"]
integer_features = bundle["integer_features"]

st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="centered"
)

st.title("🏦 Credit Risk Prediction")

st.markdown("""
<p style="font-size:16px">
Built with <b>Scikit-learn</b>, <b>XGBoost</b> & <b>Streamlit</b> — by Kuldii Project
</p>

<p style="font-size:14px">
This app predicts the credit risk (good/bad) for a loan applicant based on various financial and personal attributes.<br>
✅ Select a model<br>
📝 Enter applicant details<br>
🔮 Get the prediction and class probabilities<br>
📂 <strong>Dataset:</strong> <a href="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data" target="_blank">
Statlog (German Credit Data)</a> from UCI Machine Learning Repository.
</p>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    st.subheader("📝 Input Applicant Features")

    cols = list(feature_labels.values())
    input_values = []

    for idx in range(0, len(cols), 3):
        columns = st.columns(3)
        for j in range(3):
            if idx + j >= len(cols):
                break
            col = cols[idx + j]
            label = col

            if col in cat_cols:
                # Categorical field → 🔘
                val = columns[j].selectbox(
                    f"🔘 {label}",
                    le_dict[col].classes_.tolist(),
                    index=0
                )
            elif col in integer_features:
                # Integer field → 🔢
                stats = feature_stats.get(col, {'min': 0, 'max': 10, 'mean': 5})
                min_val = int(stats['min'])
                max_val = int(stats['max'])
                mean_val = int(round(stats['mean']))
                val = columns[j].number_input(
                    f"🔢 {label}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=1
                )
            else:
                # Numeric (float) field → 📊
                stats = feature_stats.get(col, {'min': 0.0, 'max': 10.0, 'mean': 5.0})
                min_val = float(stats['min'])
                max_val = float(stats['max'])
                mean_val = float(stats['mean'])
                val = columns[j].slider(
                    f"📊 {label}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )
            input_values.append(val)

    model_name = st.selectbox(
        "✨ Select Classification Model",
        list(models.keys()),
        index=0
    )

    submitted = st.form_submit_button("🚀 Predict Credit Risk")

if submitted:
    input_dict = {}
    for i, col in enumerate(feature_labels.values()):
        val = input_values[i]
        if col in integer_features:
            val = int(round(val)) if val is not None else int(round(feature_stats.get(col, {}).get('mean', 0)))
        elif col in cat_cols:
            if val not in le_dict[col].classes_:
                val = le_dict[col].classes_[0]
        else:
            val = float(val) if val is not None else float(feature_stats.get(col, {}).get('mean', 0))
        input_dict[col] = val

    input_df = pd.DataFrame([input_dict])

    for col in cat_cols:
        le = le_dict[col]
        input_df[col] = le.transform([input_df[col][0]])

    if num_cols:
        input_df[num_cols] = scaler.transform(input_df[num_cols])

    model = models[model_name]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    pred_label_num = le_target.inverse_transform([pred])[0]
    label_map = {'1': 'Good', '2': 'Bad'}
    pred_label = label_map.get(str(pred_label_num), str(pred_label_num))

    proba_str = "\n".join([
        f"{label_map.get(str(le_target.inverse_transform([i])[0]), str(le_target.inverse_transform([i])[0]))}: {p:.2%}"
        for i, p in enumerate(proba)
    ])

    st.success(f"💳 **Predicted Credit Risk:** {pred_label}\n\n🔎 **Probabilities:**\n{proba_str}\n\n🧮 *(Model: {model_name})*")
