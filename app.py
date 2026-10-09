from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


# Load the saved model package
with open("fraud_detection_model.pkl", "rb") as file:
    model_package = pickle.load(file)


# Get the saved objects
model = model_package["model"]
preprocessor = model_package["preprocessor"]
amount_upper_bound = model_package["amount_upper_bound"]


# Create FastAPI application
app = FastAPI(
    title="Fraud Detection API",
    description="API for predicting fraudulent transactions",
    version="1.0"
)


# Define the transaction input
class Transaction(BaseModel):
    step: int
    age: str
    gender: str
    merchant: str
    category: str
    amount: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    # Convert the input into a DataFrame
    data = pd.DataFrame([{
        "step": transaction.step,
        "age": transaction.age,
        "gender": transaction.gender,
        "merchant": transaction.merchant,
        "category": transaction.category,
        "amount": transaction.amount
    }])


    # Feature Engineering
    data["hour_cycle"] = data["step"] % 24

    data["amount_outlier"] = (
        data["amount"] > amount_upper_bound
    ).astype(int)


    # Apply the same preprocessing used during training
    data_processed = preprocessor.transform(data)


    # Make prediction
    prediction = model.predict(data_processed)[0]


    # Convert prediction to readable label
    if prediction == 1:
        label = "Fraud"
    else:
        label = "Not Fraud"


    # Return prediction
    return {
        "prediction": int(prediction),
        "label": label
    }