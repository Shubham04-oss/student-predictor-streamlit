import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("student_model.pkl")
columns = joblib.load("feature_columns.pkl")

# Page configuration
st.set_page_config(page_title="🎓 Student Math Score Predictor", layout="centered")

# Custom CSS for beauty & smoothness
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        max-width: 700px;
        margin: auto;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    h1 {
        text-align: center;
        color: #2e86de;
        font-family: 'Segoe UI', sans-serif;
    }
    .feedback {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        text-align: center;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# App container
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<h1>📘 Student Math Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("Use academic and demographic details to predict a student's **math score** using a trained machine learning model.")

with st.form("prediction_form"):
    st.markdown("### 📄 Student Details")

    reading = st.slider("📖 Reading Score", 0, 100, 70)
    writing = st.slider("✍️ Writing Score", 0, 100, 70)
    gender = st.selectbox("🚻 Gender", ["male", "female"])
    lunch = st.selectbox("🍱 Lunch Type", ["standard", "free/reduced"])
    prep = st.selectbox("📚 Test Preparation", ["none", "completed"])
    race = st.selectbox("🧬 Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    education = st.selectbox(
        "🎓 Parental Level of Education",
        [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ]
    )

    submitted = st.form_submit_button("🔍 Predict Math Score")

if submitted:
    # Prepare input data
    input_dict = {
        'reading score': reading,
        'writing score': writing,
        f'gender_{gender}': 1,
        f'lunch_{lunch}': 1,
        f'test preparation course_{prep}': 1,
        f'race/ethnicity_{race}': 1,
        f'parental level of education_{education}': 1,
    }

    final_input = {col: input_dict.get(col, 0) for col in columns}
    input_df = pd.DataFrame([final_input])

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_rounded = round(prediction, 2)

    # Display
    st.markdown("### 🎯 Predicted Math Score")
    st.progress(int(min(prediction_rounded, 100)))

    if prediction >= 90:
        feedback = f"🔥 Excellent! The predicted math score is **{prediction_rounded}**"
        color = "#d4edda"
    elif prediction >= 70:
        feedback = f"📘 Good Job! Predicted score: **{prediction_rounded}**"
        color = "#fff3cd"
    else:
        feedback = f"⚠️ Needs Improvement. Predicted score: **{prediction_rounded}**"
        color = "#f8d7da"

    st.markdown(
        f"<div class='feedback' style='background-color: {color};'>{feedback}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
