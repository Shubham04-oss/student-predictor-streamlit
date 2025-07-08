import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("student_model.pkl")
columns = joblib.load("feature_columns.pkl")

# Page config
st.set_page_config(page_title="📘 Vibrant Student Predictor", layout="centered")

# Vibrant Glassmorphic UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body {
    height: 100%;
    margin: 0;
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    background-attachment: fixed;
}

.main-container {
    background: rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.3);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 3rem;
    margin: 3rem auto;
    max-width: 800px;
    color: #fff;
}

h1 {
    text-align: center;
    font-size: 2.8rem;
    margin-bottom: 1rem;
    color: #fff;
}

.stSlider > div > div > div {
    background-color: #ffffff44 !important;
}

.stButton>button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    transition: 0.3s ease-in-out;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.feedback {
    font-size: 1.4rem;
    font-weight: 600;
    padding: 1.2rem;
    text-align: center;
    border-radius: 12px;
    margin-top: 1.5rem;
    box-shadow: 0 0 25px rgba(255, 255, 255, 0.2);
}

input, select {
    background-color: #ffffff33 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# Start main app container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("<h1>🔮 Student Math Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("Enter the student's info to predict their math score using a trained ML model.")

# Input form
with st.form("prediction_form"):
    st.markdown("### 📝 Enter Student Details")

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

    submitted = st.form_submit_button("🔍 Predict")

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

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction_rounded = round(prediction, 2)

    # Progress bar
    st.markdown("### 🎯 Predicted Math Score")
    st.progress(int(min(prediction_rounded, 100)))

    # Feedback styling
    if prediction >= 90:
        feedback = f"💥 Outstanding! Predicted math score: **{prediction_rounded}**"
        bg = "#32CD32"
    elif prediction >= 70:
        feedback = f"🌟 Great job! Predicted math score: **{prediction_rounded}**"
        bg = "#FFA500"
    else:
        feedback = f"🧠 Needs focus. Predicted math score: **{prediction_rounded}**"
        bg = "#FF4444"

    st.markdown(
        f"<div class='feedback' style='background-color: {bg}; color: white;'>{feedback}</div>",
        unsafe_allow_html=True
    )

# End container
st.markdown('</div>', unsafe_allow_html=True)
