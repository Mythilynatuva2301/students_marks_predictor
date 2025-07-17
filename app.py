import streamlit as st
import joblib

# Load the trained model
model = joblib.load("student_mark_model.pkl")

# Streamlit UI
st.title("📚 Student Marks Predictor")
st.write("Enter how many hours you studied, and I'll predict your marks!")

# Input
hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.5)

# Button
if st.button("Predict"):
    result = model.predict([[hours]])
    st.success(f"🎯 Predicted Marks: {result[0]:.2f}")
