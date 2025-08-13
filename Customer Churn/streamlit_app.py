import streamlit as st
import pandas as pd
import requests
import json
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("🔍 Customer Churn Prediction")
st.write("Predict if a customer is likely to churn using ML models.")

# Form for input
with st.form("churn_form"):
    st.subheader("📋 Customer Information")

    credit_score = st.slider("Credit Score", 300, 900, 650)
    gender = st.radio("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Account Balance", min_value=0.0, value=60000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", min_value=0.0, value=75000.0)
    geography = st.selectbox("Geography", ["Germany", "Spain", "France"])

    model_choice = st.selectbox("Model to Use", ["Logistic Regression", "Random Forest", "XGBoost"])

    submit = st.form_submit_button("Predict")

if submit:
    # Encode values
    gender_val = 1 if gender == "Male" else 0
    has_card_val = 1 if has_card == "Yes" else 0
    is_active_val = 1 if is_active == "Yes" else 0
    geo_ger = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Prepare payload
    payload = {
        "CreditScore": credit_score,
        "Gender": gender_val,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card_val,
        "IsActiveMember": is_active_val,
        "EstimatedSalary": salary,
        "Geography_Germany": geo_ger,
        "Geography_Spain": geo_spain
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=payload)
        result = response.json()

        if response.status_code == 200:
            st.success("✅ Prediction Successful!")

            model_map = {
                "Logistic Regression": "logistic_regression",
                "Random Forest": "random_forest",
                "XGBoost": "xgboost"
            }
            selected_model = model_map[model_choice]
            prediction = result["predictions"][selected_model]

            st.markdown(f"### 🔮 Prediction ({model_choice}):")
            st.markdown(f"**{'Churn' if prediction == 1 else 'No Churn'}**")

            st.subheader("📊 SHAP Explanation (Top 5 Features)")
            shap_df = pd.DataFrame(result["shap_explanation"])
            st.dataframe(shap_df)

            st.bar_chart(data=shap_df.set_index("feature"))

            # Save to CSV
            history_row = payload.copy()
            history_row["Prediction_Model"] = model_choice
            history_row["Prediction"] = prediction
            for item in result["shap_explanation"]:
                history_row[f"SHAP_{item['feature']}"] = item["shap_value"]

            history_df = pd.DataFrame([history_row])
            file_exists = os.path.exists("prediction_history.csv")
            history_df.to_csv("prediction_history.csv", mode='a', header=not file_exists, index=False)
            st.success("📝 Prediction saved to `prediction_history.csv`")

        else:
            st.error(f"❌ API Error: {result}")

    except Exception as e:
        st.error(f"🚫 Request failed: {e}")
