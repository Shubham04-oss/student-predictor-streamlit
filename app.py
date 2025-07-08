import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("student_model.pkl")
columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Student Math Score Predictor", layout="centered")

st.title("📘 Student Math Score Predictor")
st.markdown("Enter student details below to predict their **Math Score** 🎯")

# Input form
with st.form("prediction_form"):
    reading_score = st.slider("📖 Reading Score", 0, 100, 70)
    writing_score = st.slider("✍️ Writing Score", 0, 100, 70)

    gender = st.selectbox("Gender", ["male", "female"])
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    prep = st.selectbox("Test Preparation Course", ["none", "completed"])
    race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ]
    )

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    # Build input dict
    input_dict = {
        'reading score': reading_score,
        'writing score': writing_score,
        f'gender_{gender}': 1,
        f'lunch_{lunch}': 1,
        f'test preparation course_{prep}': 1,
        f'race/ethnicity_{race}': 1,
        f'parental level of education_{education}': 1,
    }

    # Fill missing dummy features with 0
    input_vector = {col: input_dict.get(col, 0) for col in columns}
    input_df = pd.DataFrame([input_vector])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Math Score: **{prediction:.2f}** / 100")
