import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("best_rf_model.pkl")

# -------------------------
# Load dataset to rebuild encoders
# -------------------------
data = pd.read_csv("Telco-Customer-Churn.csv")

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Target column
target = "Churn"

# Build LabelEncoders
encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    if col != target:  # don't encode target here
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📊 Telco Customer Churn Prediction App")

st.sidebar.header("Enter Customer Details")

def user_input():
    input_data = {}
    for col in data.drop(columns=[target]).columns:  # features only
        if col in encoders:  # categorical
            val = st.sidebar.selectbox(f"{col}", encoders[col].classes_)
            input_data[col] = encoders[col].transform([val])[0]
        else:  # numeric
            val = st.sidebar.number_input(f"{col}", value=float(data[col].mean()))
            input_data[col] = val
    return pd.DataFrame([input_data])

input_df = user_input()

st.subheader("🔍 Encoded Customer Input")
st.write(input_df)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📢 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {probability:.2f})")
