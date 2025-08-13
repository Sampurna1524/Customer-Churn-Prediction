from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Customer Churn Prediction API")

# Load models
try:
    log_model = joblib.load("models/logistic_model.pkl")
    rf_model = joblib.load("models/random_forest_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    shap_explainer = shap.Explainer(xgb_model)
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

# Define input schema
class CustomerData(BaseModel):
    CreditScore: int
    Gender: int                 # 0 = Female, 1 = Male
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int              # 0 = No, 1 = Yes
    IsActiveMember: int         # 0 = No, 1 = Yes
    EstimatedSalary: float
    Geography_Germany: int      # 0 or 1
    Geography_Spain: int        # 0 or 1

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Customer Churn Prediction API!"}

# Prediction endpoint
@app.post("/predict/")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    try:
        # Model predictions
        pred_log = int(log_model.predict(input_df)[0])
        pred_rf = int(rf_model.predict(input_df)[0])
        pred_xgb = int(xgb_model.predict(input_df)[0])

        # SHAP values for XGBoost
        shap_values = shap_explainer(input_df)
        explanation = [
            {
                "feature": feature,
                "shap_value": float(round(value, 4))
            }
            for feature, value in sorted(
                zip(input_df.columns, shap_values[0].values),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 impactful features
        ]

        return {
            "predictions": {
                "logistic_regression": pred_log,
                "random_forest": pred_rf,
                "xgboost": pred_xgb
            },
            "shap_explanation": explanation
        }

    except Exception as e:
        print("Error during prediction:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
