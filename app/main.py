import streamlit as st
import pandas as pd
import os
import joblib

# =======================
# Configuration
# =======================
MODEL_SOURCE = 'app\model\svm_model.pkl'
# You can set MODEL_SOURCE to:
# - MLflow registry: "models:/SVMTextClassifier/latest"
# - Local path: "./models/my_model"
# - URL: "https://path/to/model"

# =======================
# Load Model
# =======================
@st.cache_resource(show_spinner=True)
def load_model(source):
    return joblib.load(source)

try:
    model = load_model(MODEL_SOURCE)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_SOURCE}: {e}")
    st.stop()

# =======================
# Prediction Mapping
# =======================
label_map = {0: "Human", 1: "AI"}

# =======================
# Streamlit UI
# =======================
st.title("AI vs Human Text Classifier")
st.write("Enter a text below and find out whether it was written by a human or AI:")

user_input = st.text_area("Your Text Here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting!")
    else:
        try:
            # Adjust if your model expects a specific column name
            input_df = pd.DataFrame({"clean_text": [user_input]})
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {label_map[int(prediction)]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
