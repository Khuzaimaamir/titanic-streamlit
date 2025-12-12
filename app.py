import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ------------------------------
# Folder Path
# ------------------------------
folder_path = r"E:\100 Projects\1\titanic-streamlit\\"

# ------------------------------
# Load Models
# ------------------------------
rf = pickle.load(open(folder_path + "rf_model.pkl", "rb"))
xgb = pickle.load(open(folder_path + "xgb_model.pkl", "rb"))
lgbm = pickle.load(open(folder_path + "lgbm_model.pkl", "rb"))

models_dict = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgbm
}

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction App")
st.write("Predict survival probability based on passenger details")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Passenger Details & Model Selection")

# Dropdown to select model
model_choice = st.sidebar.selectbox("Choose Model", list(models_dict.keys()))

pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses", 0, 8, 0)
parch = st.sidebar.number_input("Number of Parents/Children", 0, 6, 0)
fare = st.sidebar.number_input("Ticket Fare", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C","Q","S"])
cabin = st.sidebar.text_input("Cabin (Optional)")

# ------------------------------
# Feature Engineering
# ------------------------------
family_size = sibsp + parch + 1
alone_check = 1 if family_size == 1 else 0
fare_per_person = fare / family_size
is_rich = 1 if pclass == 1 else 0
has_deck = 1 if cabin.strip() != "" else 0

sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# ------------------------------
# Build Input DataFrame
# ------------------------------
input_data = pd.DataFrame({
    'pclass':[pclass],
    'age':[age],
    'sibsp':[sibsp],
    'parch':[parch],
    'fare':[fare],
    'family_size':[family_size],
    'alone_check':[alone_check],
    'fare_per_person':[fare_per_person],
    'is_rich':[is_rich],
    'has_deck':[has_deck],
    'sex_male':[sex_male],
    'embarked_Q':[embarked_Q],
    'embarked_S':[embarked_S]
})

# ------------------------------
# Scale numeric columns for RF, XGB, LGBM
# ------------------------------
num_cols = ['age','sibsp','parch','fare','family_size','fare_per_person']
scaler = pickle.load(open(folder_path + "scaler.pkl", "rb"))
input_data[num_cols] = scaler.transform(input_data[num_cols])

# ------------------------------
# Select model for prediction & feature importance
# ------------------------------
model = models_dict[model_choice]

# ------------------------------
# Prediction
# ------------------------------
st.header(f"🔮 Prediction using {model_choice}")
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"🟢 Survived! Probability: {prob*100:.2f}%")
    else:
        st.error(f"🔴 Did NOT Survive. Probability: {prob*100:.2f}%")

# ------------------------------
# Feature Importance
# ------------------------------
st.header(f"📊 Feature Importance ({model_choice})")
importances = model.feature_importances_

feat_names = input_data.columns
feat_imp = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
fig = px.bar(feat_imp, x='Feature', y='Importance', color='Importance', color_continuous_scale='Viridis')
st.plotly_chart(fig, use_container_width=True)
