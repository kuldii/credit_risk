import joblib
import pandas as pd
import gradio as gr

# Load semua asset
bundle = joblib.load("models/model_bundle.pkl")

# Unpack
models = bundle["models"]
le_dict = bundle["le_dict"]
le_target = bundle["le_target"]
scaler = bundle["scaler"]
feature_labels = bundle["feature_labels"]
feature_stats = bundle["feature_stats"]
cat_cols = bundle["cat_cols"]
num_cols = bundle["num_cols"]
integer_features = bundle["integer_features"]

# Prediction function
def predict_credit_risk_blocks(selected_model, *feature_values):
    input_dict = {}
    for i, col in enumerate(feature_labels.values()):
        val = feature_values[i]
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
    model = models[selected_model]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    pred_label_num = le_target.inverse_transform([pred])[0]
    label_map = {'1': 'Good', '2': 'Bad'}
    pred_label = label_map.get(str(pred_label_num), str(pred_label_num))
    proba_str = "\n".join([
        f"{label_map.get(str(le_target.inverse_transform([i])[0]), str(le_target.inverse_transform([i])[0]))}: {p:.2%}"
        for i, p in enumerate(proba)
    ])
    return f"💳 Predicted Credit Risk: {pred_label}\n\nProbabilities:\n{proba_str}\n\n(Model: {selected_model})"

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("""
    <h2>🏦 Credit Risk Prediction</h2>
    <p style=\"font-size: 16px;\">
    Built with Scikit-learn, XGBoost & Gradio — by Kuldii Project
    </p>
    <p style=\"font-size: 14px;\">
    This app predicts the credit risk (good/bad) for a loan applicant based on various financial and personal attributes.<br>
    ✅ Select a model<br>
    📝 Enter applicant details<br>
    🔮 Get the prediction and class probabilities<br>
    📂 <strong>Dataset:</strong> <a href=\"https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data\" target=\"_blank\">
    Statlog (German Credit Data)</a> from UCI Machine Learning Repository.
    </p>
    """)
    model_choice = gr.Dropdown(
        choices=list(models.keys()),
        label="✨ Select Classification Model",
        value=list(models.keys())[0]
    )
    input_elems = []
    cols = list(feature_labels.values())
    for idx in range(0, len(cols), 3):
        with gr.Row():
            for j in range(3):
                if idx + j >= len(cols):
                    break
                col = cols[idx + j]
                label = col
                if col in cat_cols:
                    comp = gr.Dropdown(
                        choices=le_dict[col].classes_.tolist(),
                        label=f"🔘 {label}",
                        value=le_dict[col].classes_[0]
                    )
                elif col in integer_features:
                    stats = feature_stats.get(col, {'min': 0, 'max': 10, 'mean': 5})
                    min_val = int(stats['min'])
                    max_val = int(stats['max'])
                    mean_val = int(round(stats['mean']))
                    comp = gr.Number(
                        label=f"🔢 {label}",
                        value=mean_val,
                        minimum=min_val,
                        maximum=max_val,
                        step=1
                    )
                else:
                    stats = feature_stats.get(col, {'min': 0.0, 'max': 10.0, 'mean': 5.0})
                    min_val = float(stats['min'])
                    max_val = float(stats['max'])
                    mean_val = float(stats['mean'])
                    comp = gr.Slider(
                        minimum=min_val,
                        maximum=max_val,
                        value=mean_val,
                        label=f"📊 {label}"
                    )
                input_elems.append(comp)
    predict_btn = gr.Button("🚀 Predict Credit Risk")
    output = gr.Textbox(label="🔎 Prediction Result", lines=5)
    predict_btn.click(
        fn=predict_credit_risk_blocks,
        inputs=[model_choice] + input_elems,
        outputs=output
    )

demo.launch(server_name="0.0.0.0", root_path="/credit_risk", server_port=9003)