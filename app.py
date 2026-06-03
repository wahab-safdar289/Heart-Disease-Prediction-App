import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction by WS AI Labs")
st.write("Enter patient details below")

# Inputs
age = st.slider("Age", 18, 100, 40)

gender = st.selectbox("Gender", ["Male", "Female"])

bp = st.number_input("Blood Pressure", 80, 250, 120)

cholesterol = st.number_input("Cholesterol Level", 100, 500, 200)

exercise_habits = st.slider("Exercise Habits (0-10)", 0, 10, 5)

smoker = st.selectbox("Smoker", ["No", "Yes"])

family_history = st.selectbox(
    "Family History of Heart Disease",
    ["No", "Yes"]
)

diabetes = st.selectbox(
    "Diabetes",
    ["No", "Yes"]
)

bmi = st.number_input(
    "BMI",
    min_value=15.0,
    max_value=50.0,
    value=25.0
)

high_bp = st.selectbox(
    "High Blood Pressure",
    ["No", "Yes"]
)

low_hdl = st.selectbox(
    "Low HDL Cholesterol",
    ["No", "Yes"]
)

high_ldl = st.selectbox(
    "High LDL Cholesterol",
    ["No", "Yes"]
)

alcohol = st.slider(
    "Alcohol Consumption (0-10)",
    0, 10, 2
)

stress = st.slider(
    "Stress Level (0-10)",
    0, 10, 5
)

sleep = st.slider(
    "Sleep Hours",
    0, 12, 7
)

sugar = st.slider(
    "Sugar Consumption (0-10)",
    0, 10, 5
)

triglyceride = st.number_input(
    "Triglyceride Level",
    50, 500, 150
)

fasting_bs = st.number_input(
    "Fasting Blood Sugar",
    50, 300, 100
)

crp = st.number_input(
    "CRP Level",
    min_value=0.0,
    max_value=20.0,
    value=2.0
)

homocysteine = st.number_input(
    "Homocysteine Level",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)



input_df = pd.DataFrame([{
        'Age': age,
        'Is_Female': 1 if gender == "Female" else 0,
        'Blood Pressure': bp,
        'Cholesterol Level': cholesterol,
        'Exercise Habits': exercise_habits,
        'Is_Smoker': 1 if smoker == "Yes" else 0,
        'Family_Has_Heart_Disease': 1 if family_history == "Yes" else 0,
        'Has_Diabetes': 1 if diabetes == "Yes" else 0,
        'BMI': bmi,
        'Has_High_Blood_Pressure': 1 if high_bp == "Yes" else 0,
        'Has_Low_HDL_Cholesterol': 1 if low_hdl == "Yes" else 0,
        'Has_High_LDL_Cholesterol': 1 if high_ldl == "Yes" else 0,
        'Alcohol Consumption': alcohol,
        'Stress Level': stress,
        'Sleep Hours': sleep,
        'Sugar Consumption': sugar,
        'Triglyceride Level': triglyceride,
        'Fasting Blood Sugar': fasting_bs,
        'CRP Level': crp,
        'Homocysteine Level': homocysteine
}])

scaled_columns = [
        'Age',
        'Blood Pressure',
        'Cholesterol Level',
        'Exercise Habits',
        'Alcohol Consumption',
        'Stress Level',
        'Sleep Hours',
        'Sugar Consumption',
        'Triglyceride Level',
        'Fasting Blood Sugar',
        'CRP Level',
        'Homocysteine Level',
        'BMI'
] 

if st.button("Predict"): 
        input_scaled = input_df.copy()

        input_scaled[scaled_columns] = scaler.transform(input_scaled[scaled_columns])
              
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
           st.error("⚠️ High Risk of Heart Disease")
        else:
           st.success("✅ Low Risk of Heart Disease")