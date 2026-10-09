# Fraud Detection Using Machine Learning

A machine learning project for detecting fraudulent financial transactions in an imbalanced dataset.

## Project Overview

The goal of this project is to classify transactions as fraudulent or non-fraudulent.

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Class imbalance handling
- Multiple machine learning experiments
- Hyperparameter tuning
- Model evaluation
- Feature importance
- FastAPI prediction API

## Models Tested

- Logistic Regression with Class Weight
- Logistic Regression with SMOTENC
- Random Forest
- Gradient Boosting
- CatBoost

The strongest candidate models were further evaluated and tuned using RandomizedSearchCV.

## Final Model

Tuned Gradient Boosting was selected based on validation F1-score.

Final test performance:

- Precision: 0.8635
- Recall: 0.7646
- F1-score: 0.8110

## API

The trained model is exposed through a FastAPI application.

Run locally:

uvicorn app:app --reload

Then open the API documentation at:

http://127.0.0.1:8000/docs

## Future Improvements

- Decision threshold tuning
- Additional feature engineering
- More robust validation strategies
- Model deployment
- Model monitoring