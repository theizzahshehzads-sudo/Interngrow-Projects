"""
AI Email Classification System — Interactive Dashboard
--------------------------------------------------------
Streamlit application providing:
  - User Authentication (login screen)
  - Interactive multi-page Dashboard
  - Single & batch (CSV) prediction with decision output
  - Model performance visualization
  - Robust error handling throughout

Run locally:
    streamlit run app.py

Requires model.pkl, vectorizer.pkl and metrics.json (produced by the
companion Colab notebook) to be present in the same directory.
"""

import os
import re
import json
import string
import joblib
import pandas as pd
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="AI Email Classification System",
    page_icon="📧",
    layout="wide",
)


DEMO_USERS = {
    "admin": "admin123",
    "user": "user123",
}


def login_screen():
    st.markdown("## 🔐 AI Email Classification System — Login")
    st.caption("Demo credentials: `admin` / `admin123`  or  `user` / `user123`")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            if username in DEMO_USERS and DEMO_USERS[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")
        except Exception as e:
            st.error(f"⚠️ Login error: {e}")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# ----------------------------------------------------------------------
# --- Load model artifacts (with error handling) ------------------------
# ----------------------------------------------------------------------
# Resolve paths relative to THIS script's own folder, not the process's
# working directory. This matters because on Streamlit Cloud (and some
# other hosts) the app can be launched from the repo root even when
# app.py lives in a subfolder, which would otherwise break plain
# filenames like "model.pkl".
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(filename):
    return os.path.join(APP_DIR, filename)


@st.cache_resource
def load_artifacts():
    errors = []
    model, vectorizer, metrics = None, None, {}
    try:
        model = joblib.load(_path("model.pkl"))
    except Exception as e:
        errors.append(f"Could not load model.pkl: {e}")
    try:
        vectorizer = joblib.load(_path("vectorizer.pkl"))
    except Exception as e:
        errors.append(f"Could not load vectorizer.pkl: {e}")
    try:
        with open(_path("metrics.json"), "r") as f:
            metrics = json.load(f)
    except Exception as e:
        errors.append(f"Could not load metrics.json: {e}")
    return model, vectorizer, metrics, errors


model, vectorizer, metrics, load_errors = load_artifacts()

# ----------------------------------------------------------------------
# --- Preprocessing (must mirror the notebook's clean_text) -------------
# ----------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for resource in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    STOP_WORDS = set()
    LEMMATIZER = None


def clean_text(text):
    """Clean raw email text — mirrors preprocessing used during training."""
    try:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.strip()
        tokens = text.split()
        if LEMMATIZER:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
        else:
            tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return " ".join(tokens)
    except Exception:
        return ""


def predict_email(text):
    """Predict & Decision Output for a single email string."""
    if model is None or vectorizer is None:
        return {"error": "Model artifacts are not loaded. Run the notebook and place model.pkl / vectorizer.pkl here."}
    if not isinstance(text, str) or text.strip() == "":
        return {"error": "Input text is empty."}
    try:
        cleaned = clean_text(text)
        if cleaned.strip() == "":
            return {"label": "ham", "confidence": 0.5, "top_terms": [], "note": "No meaningful tokens after cleaning."}
        vec_input = vectorizer.transform([cleaned])
        pred = model.predict(vec_input)[0]
        proba = model.predict_proba(vec_input)[0]
        confidence = float(max(proba))
        label = "spam" if pred == 1 else "ham"

        feature_names = np.array(vectorizer.get_feature_names_out())
        row = vec_input.toarray()[0]
        top_idx = row.argsort()[::-1][:5]
        top_terms = [feature_names[i] for i in top_idx if row[i] > 0]

        return {"label": label, "confidence": round(confidence, 4), "top_terms": top_terms}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}


# ----------------------------------------------------------------------
# --- Sidebar navigation --------------------------------------------------
# ----------------------------------------------------------------------
st.sidebar.title("📧 Email Classifier")
st.sidebar.markdown(f"Logged in as **{st.session_state.get('username', 'user')}**")
if st.sidebar.button("Log out"):
    st.session_state["authenticated"] = False
    st.rerun()

page = st.sidebar.radio(
    "Navigate",
    ["Single Prediction", "Batch Prediction (CSV)", "Model Performance", "About / Documentation"],
)

if load_errors:
    for err in load_errors:
        st.sidebar.error(err)

# ----------------------------------------------------------------------
# --- Page: Single Prediction --------------------------------------------
# ----------------------------------------------------------------------
if page == "Single Prediction":
    st.title("✉️ Classify a Single Email")
    st.write("Paste the content of an email below and classify it as **Spam** or **Ham** (legitimate).")

    email_text = st.text_area("Email content", height=200, placeholder="Paste email text here...")

    if st.button("🔍 Classify Email", type="primary"):
        with st.spinner("Analyzing..."):
            result = predict_email(email_text)

        if "error" in result:
            st.error(result["error"])
        else:
            label = result["label"]
            confidence = result["confidence"]

            if label == "spam":
                st.error(f"🚨 **SPAM** detected — confidence {confidence*100:.1f}%")
            else:
                st.success(f"✅ **HAM** (legitimate) — confidence {confidence*100:.1f}%")

            st.progress(confidence)

            if result.get("top_terms"):
                st.caption("Top contributing terms: " + ", ".join(f"`{t}`" for t in result["top_terms"]))

# ----------------------------------------------------------------------
# --- Page: Batch Prediction ----------------------------------------------
# ----------------------------------------------------------------------
elif page == "Batch Prediction (CSV)":
    st.title("📂 Batch Email Classification")
    st.write("Upload a CSV file with a column named **`message`** containing one email per row.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            if "message" not in batch_df.columns:
                st.error("The CSV must contain a column named 'message'.")
            else:
                with st.spinner("Classifying all messages..."):
                    labels, confidences = [], []
                    for msg in batch_df["message"]:
                        res = predict_email(msg)
                        if "error" in res:
                            labels.append("error")
                            confidences.append(0.0)
                        else:
                            labels.append(res["label"])
                            confidences.append(res["confidence"])
                    batch_df["predicted_label"] = labels
                    batch_df["confidence"] = confidences

                st.success(f"Classified {len(batch_df)} messages.")
                st.dataframe(batch_df, use_container_width=True)

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download results as CSV", csv_out, "predictions.csv", "text/csv")

                st.bar_chart(batch_df["predicted_label"].value_counts())
        except Exception as e:
            st.error(f"⚠️ Could not process the uploaded file: {e}")

# ----------------------------------------------------------------------
# --- Page: Model Performance ----------------------------------------------
# ----------------------------------------------------------------------
elif page == "Model Performance":
    st.title("📊 Model Performance Evaluation")

    if not metrics:
        st.warning("metrics.json not found. Run the notebook's evaluation section and place metrics.json here.")
    else:
        try:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
            col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")

            st.markdown(f"**Best model:** `{metrics['best_model']}`")

            if metrics.get("all_results"):
                st.subheader("Comparison across all trained models")
                results_df = pd.DataFrame(metrics["all_results"])
                st.dataframe(results_df, use_container_width=True)
                st.bar_chart(results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]])
        except Exception as e:
            st.error(f"⚠️ Could not display metrics: {e}")

        for img_name, caption in [
            ("confusion_matrix.png", "Confusion Matrix"),
            ("roc_curve.png", "ROC Curve"),
            ("model_comparison.png", "Model Comparison"),
            ("class_distribution.png", "Class Distribution"),
        ]:
            img_path = _path(img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, use_container_width=True)

# ----------------------------------------------------------------------
# --- Page: About / Documentation ------------------------------------------
# ----------------------------------------------------------------------
elif page == "About / Documentation":
    st.title("ℹ️ About This Project")
    st.markdown("""
### AI Email Classification System

An end-to-end machine learning system that classifies emails as **Spam** or **Ham** (legitimate),
built to satisfy the following mandatory project requirements:

- **AI Model Integration** — Naive Bayes, Logistic Regression and Linear SVM, best model auto-selected
- **Data Preprocessing** — cleaning, tokenization, stopword removal, lemmatization, TF-IDF features
- **Model Training** — trained and cross-validated in the companion Colab notebook
- **User Authentication** — this login screen
- **Interactive Dashboard** — the multi-page app you're using now
- **Prediction & Decision Output** — label, confidence score, and top contributing terms
- **Performance Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, ROC curve
- **Error Handling** — every stage (loading, cleaning, predicting, file uploads) is wrapped safely
- **Professional Documentation** — see `README.md` in the project repository
- **Deployment** — this app is deployable to Streamlit Cloud, Hugging Face Spaces, Render, or Azure

**Tech stack:** Python, scikit-learn, NLTK, pandas, Streamlit.
""")
