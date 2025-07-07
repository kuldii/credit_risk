# Credit Risk Prediction App

A professional, production-ready machine learning app for predicting credit risk (good/bad) using the Statlog (German Credit Data) dataset from the UCI Machine Learning Repository. Built with robust preprocessing, user-friendly feature mapping, multiple classification models, and a modern Gradio UI. All models and preprocessing objects are saved for easy deployment.

---

## 🚀 Features

- **Robust Preprocessing**: Categorical encoding, numeric scaling, user-friendly feature mapping
- **Multiple Classification Models**: Logistic Regression, Random Forest, XGBoost (with hyperparameter tuning)
- **Rich Visualizations & EDA**: Class distribution, histograms, boxplots, correlation heatmap, feature statistics
- **Interactive Gradio UI**: Model selection, grouped input fields (dropdowns, sliders, numbers), clear prediction output, modern layout
- **Deployment-Ready**: All models and preprocessing objects are saved for instant prediction in the app

---

## 🏗️ Project Structure

```
credit_risk/
├── app.py                  # Gradio app for prediction (production-ready)
├── credit_risk_app.ipynb   # Full EDA, modeling, and training notebook
├── models/
│   ├── model_bundle.pkl    # Trained models and preprocessing objects (joblib)
│   └── ...
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Data & Preprocessing

- **Dataset**: [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Preprocessing**:
  - User-friendly feature names from dataset metadata
  - Label encoding for categoricals
  - Standardization of numeric features
  - Feature statistics (min, max, mean) for UI defaults

---

## 🧠 Models

- **Logistic Regression** (with GridSearchCV)
- **Random Forest Classifier** (with GridSearchCV)
- **XGBoost Classifier** (with GridSearchCV)

All models are trained and saved for instant prediction in the app.

---

## 🖥️ Gradio App

- **Dropdowns, sliders, and numbers** for all features (custom min/max for each)
- **Model selection** dropdown
- **Prediction output**: Credit risk (Good/Bad) and class probabilities
- **Production config**: Runs on `0.0.0.0:9003` with `/credit_risk` root path (for Docker/deployment)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <this-repo-url>
cd credit_risk
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train Models
- All models and preprocessing objects are pre-trained and saved in `models/`.
- To retrain, use the notebook `credit_risk_app.ipynb` and re-export the models.

### 4. Run the App
```bash
python app.py
```
- The app will be available at `http://localhost:9003/credit_risk` by default.

---

## 🖥️ Usage

1. Open the app in your browser.
2. Input applicant features (all fields are labeled and grouped for clarity).
3. Select a classification model.
4. Click **Predict Credit Risk** to get the prediction and class probabilities.

---

## 📊 Visualizations & EDA
- See `credit_risk_app.ipynb` for:
  - Class distribution
  - Feature statistics
  - Correlation heatmap
  - Histograms, boxplots, and more

---

## 📝 Model Details
- **Preprocessing**: LabelEncoder for categoricals, StandardScaler for numerics, user-friendly feature mapping.
- **Models**: LogisticRegression, RandomForestClassifier, XGBoostClassifier (all with GridSearchCV for tuning).
- **Feature Info**: All feature statistics (min, max, mean) are saved for robust UI defaults.

---

## 📁 File Descriptions
- `app.py`: Gradio app, loads models, handles prediction and UI.
- `models/model_bundle.pkl`: Dictionary of trained models and preprocessing objects.
- `requirements.txt`: Python dependencies.
- `credit_risk_app.ipynb`: Full EDA, preprocessing, model training, and export.

---

## 🌐 Demo & Credits
- **Author**: Sandikha Rahardi (Kuldii Project)
- **Website**: https://kuldiiproject.com
- **Dataset**: [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **UI**: [Gradio](https://gradio.app/)
- **ML**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)

---

For questions or contributions, please open an issue or pull request.
