import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Folder Path
# ------------------------------
folder_path = r"E:\100 Projects\1\titanic-streamlit\\"  # Update if needed

# ------------------------------
# Load Model & Scaler
# ------------------------------
rf = pickle.load(open(folder_path + "rf_model.pkl", "rb"))
scaler = pickle.load(open(folder_path + "scaler.pkl", "rb"))

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction App")
st.write("Predict survival probability based on passenger details")

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv(folder_path + "titanic_full.csv")
st.subheader("Dataset Sample")
st.dataframe(df.head())

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

# Scale numeric features
num_cols = ['age','sibsp','parch','fare','family_size','fare_per_person']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# ------------------------------
# Prediction
# ------------------------------
st.header("🔮 Prediction")

if st.button("Predict"):
    pred = rf.predict(input_data)[0]
    prob = rf.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"🟢 Survived! Probability: {prob*100:.2f}%")
    else:
        st.error(f"🔴 Did NOT Survive. Probability: {prob*100:.2f}%")

# ------------------------------
# Visualizations
# ------------------------------
st.header("📊 Feature Correlation")
numeric_cols = df.select_dtypes(include=np.number).columns
fig_corr = sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
st.pyplot(fig_corr.figure)

if "is_rich" in df.columns:
    st.header("💰 Survival by Wealth")
    fig_rich = px.histogram(df, x='is_rich', color='survived',
                            barmode='group',
                            labels={'is_rich':'Rich (1=Yes,0=No)', 'survived':'Survival'})
    st.plotly_chart(fig_rich, use_container_width=True)
