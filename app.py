import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("student_model.pkl")
columns = joblib.load("feature_columns.pkl")

# Page configuration
st.set_page_config(page_title="🎓 Student Math Score Predictor", layout="centered")

# Custom CSS for a colorful, minimal, and professional UI
st.markdown("""
    <style>
    body {
        background-color: #E8ECEF;
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F7FA 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        max-width: 750px;
        margin: 2rem auto;
    }
    h1 {
        text-align: center;
        color: #1F252A;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .stSlider > div > div > div > div {
        background-color: #2AB7CA;
    }
    .stSlider > div > div > div > div > div {
        background-color: #1F252A;
    }
    .stSelectbox > div > div > div {
        border: 1px solid #2AB7CA;
        border-radius: 8px;
        background-color: #FFFFFF;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2AB7CA 0%, #1F252A 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .stProgress > div > div {
        background-color: #2AB7CA;
    }
    .feedback {
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1.5rem;
        text-align: center;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-top: 1.5rem;
    }
    .stForm {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .stMarkdown > div > p {
        color: #1F252A;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App container
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<h1>📘 Student Math Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("Predict a student's **math score** using academic and demographic details with our trained machine learning model.")

with st.form("prediction_form"):
    st.markdown("### 📄 Student Details")

    reading = st.slider("📖 Reading Score", 0, 100, 70, help="Select the student's reading score")
    writing = st.slider("✍️ Writing Score", 0, 100, 70, help="Select the student's writing score")
    gender = st.selectbox("🚻 Gender", ["male", "female"], help="Select the student's gender")
    lunch = st.selectbox("🍱 Lunch Type", ["standard", "free/reduced"], help="Select the student's lunch type")
    prep = st.selectbox("📚 Test Preparation", ["none", "completed"], help="Select test preparation status")
    race = st.selectbox("🧬 Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"], help="Select race/ethnicity")
    education = st.selectbox(
        "🎓 Parental Level of Education",
        [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ],
        help="Select parental education level"
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
        color = "#28A745"
    elif prediction >= 70:
        feedback = f"📘 Good Job! Predicted score: **{prediction_rounded}**"
        color = "#FFA500"
    else:
        feedback = f"⚠️ Needs Improvement. Predicted score: **{prediction_rounded}**"
        color = "#DC3545"

    st.markdown(
        f"<div class='feedback' style='background-color: {color}; color: white;'>{feedback}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)
