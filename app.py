import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# Load Model and Scaler
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability")

# ------------------------------
# User Inputs
# ------------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Ticket Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# ------------------------------
# Feature Engineering (Same as Notebook)
# ------------------------------
family_size = sibsp + parch + 1
alone_check = 1 if family_size == 1 else 0
fare_per_person = fare / family_size
is_rich = 1 if fare > 50 else 0
has_deck = 0  # deck is dropped in our dataset

# Build DataFrame
input_data = pd.DataFrame({
    'pclass': [pclass],
    'age': [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare],
    'family_size': [family_size],
    'alone_check': [alone_check],
    'fare_per_person': [fare_per_person],
    'is_rich': [is_rich],
    'has_deck': [has_deck],
    'sex_male': [1 if sex == "male" else 0],
    'embarked_Q': [1 if embarked == "Q" else 0],
    'embarked_S': [1 if embarked == "S" else 0]
})

# Scale numerical columns
num_cols = ['age','sibsp','parch','fare','family_size','fare_per_person']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🟢 Survived with probability: {probability:.2f}")
    else:
        st.error(f"🔴 Did NOT survive. Probability of survival: {probability:.2f}")
