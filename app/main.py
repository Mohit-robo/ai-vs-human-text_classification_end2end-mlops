# app.py
import streamlit as st
import mlflow.pyfunc
import pandas as pd

# Load MLflow model directly from the registered model
# Replace "SVMTextClassifier" and version if needed
model = mlflow.pyfunc.load_model("models:/SVMTextClassifier/5")

# Mapping prediction numbers to labels
label_map = {0: "Human", 1: "AI"}

st.title("AI vs Human Text Classifier")
st.write("Enter a text below and find out whether it was written by a human or AI:")

# Text input from user
user_input = st.text_area("Your Text Here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting!")
    else:
        try:
            # MLflow model expects a list or DataFrame
            # If your model expects column 'clean_text', use DataFrame
            input_df = pd.DataFrame({"clean_text": [user_input]})
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {label_map[int(prediction)]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
