import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Load Model and Scaler
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction App")
st.write("Predict survival probability based on passenger details")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Passenger Details")

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

# Ensure numeric columns match model
num_cols = ['age','sibsp','parch','fare','family_size','fare_per_person']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# ------------------------------
# Prediction
# ------------------------------
st.header("🔮 Survival Prediction")
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.success(f"🟢 Survived! Probability: {prob*100:.2f}%")
    else:
        st.error(f"🔴 Did NOT Survive. Probability: {prob*100:.2f}%")

# ------------------------------
# EDA / Interactive Plots
# ------------------------------
st.header("📊 Titanic Dataset EDA")

# Load Titanic dataset (original Kaggle columns)
df = pd.read_csv("titanic.csv")  # Ensure titanic.csv is in same folder

tab1, tab2, tab3 = st.tabs(["Survival Analysis", "Feature Importance", "Correlation"])

with tab1:
    st.subheader("Survival by Sex")
    fig = px.histogram(df, x='Sex', color='Survived', barmode='group', 
                       labels={'Sex':'Gender','Survived':'Survival'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Survival by Class")
    fig = px.histogram(df, x='Pclass', color='Survived', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Survival by Age Group")
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=["0-12","13-20","21-40","41-60","61-80"])
    fig = px.histogram(df, x='AgeGroup', color='Survived', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Random Forest Feature Importance")
    importances = model.feature_importances_
    feat_names = input_data.columns
    feat_imp = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig = px.bar(feat_imp, x='Feature', y='Importance', color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)  # Only numeric columns
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmin=-1, zmax=1
    ))
    st.plotly_chart(fig, use_container_width=True)
