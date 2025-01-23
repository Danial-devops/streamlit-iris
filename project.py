import streamlit as st
import pandas as pd
import joblib

st.write("""
# Stroke Prediction App
This app predicts whether a person is likely to have a **stroke** based on various health parameters.
""")

st.sidebar.header('User Input Parameters')

# Function to get user input
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 18, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.sidebar.selectbox('Ever Married', ['No', 'Yes'])
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 55.0, 275.0, 100.0)
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
    smoking_status = st.sidebar.selectbox('Smoking Status', ['Unknown', 'Formerly Smoked', 'Never Smoked', 'Smokes'])
    work_type = st.sidebar.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])

    # Map inputs to numerical values if needed
    data = {'gender': 1 if gender == 'Female' else 0,
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'Residence_type_Rural': 1 if residence_type == 'Rural' else 0,
            'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
            'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
            'smoking_status_formerly smoked': 1 if smoking_status == 'Formerly Smoked' else 0,
            'smoking_status_never smoked': 1 if smoking_status == 'Never Smoked' else 0,
            'smoking_status_smokes': 1 if smoking_status == 'Smokes' else 0,
            'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
            'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
            'work_type_Private': 1 if work_type == 'Private' else 0,
            'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
            'work_type_children': 1 if work_type == 'children' else 0}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

model = joblib.load('best_logreg.pkl')  

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write('Stroke' if prediction == 1 else 'No Stroke')

st.subheader('Prediction Probability')
st.write(f"Probability of Stroke: {prediction_proba[0][1]*100:.2f}%")
st.write(f"Probability of No Stroke: {prediction_proba[0][0]*100:.2f}%")
