import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction System")

# Load model
model = pickle.load(open("heart_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.write("Enter Patient Details")

# Inputs
age = st.number_input("Age", 1, 120)

sex = st.selectbox("Gender", ["Female","Male"])

cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"]
)

trestbps = st.number_input("Resting Blood Pressure", 80, 200)

chol = st.number_input("Cholesterol", 100, 400)

thalach = st.number_input("Max Heart Rate", 60, 220)

# Predict button
if st.button("Predict Heart Risk"):

    # Convert inputs
    sex_val = 1 if sex == "Male" else 0

    cp_map = {
        "Typical Angina":0,
        "Atypical Angina":1,
        "Non-Anginal Pain":2,
        "Asymptomatic":3
    }

    cp_val = cp_map[cp]

    # Prepare data
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, thalach]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prob = model.predict_proba(input_scaled)

    risk = prob[0][1]

    st.write("Heart Disease Risk:", round(risk*100,2), "%")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk*100,
        title={'text': "Heart Risk %"},
        gauge={
            'axis': {'range':[0,100]},
            'steps':[
                {'range':[0,40],'color':'green'},
                {'range':[40,70],'color':'yellow'},
                {'range':[70,100],'color':'red'}
            ]
        }
    ))

    st.plotly_chart(fig)

    # Result
    if risk > 0.6:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")