# app.py
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()
model = joblib.load("fraud_model.pkl")


@app.post("/predict")
def predict(user_id: int, current_amount: float):
    # 1. Connect to the DB we created in Step 1
    conn = sqlite3.connect("production_db.db")

    # 2. Fetch history for this specific user to calculate features
    query = f"SELECT amount FROM transactions WHERE user_id = {user_id}"
    history = pd.read_sql(query, conn)
    conn.close()

    # 3. RE-CALCULATE the feature (Logic Duplication!)
    # If you change this logic in train.py, you MUST remember to change it here too.
    if not history.empty:
        user_avg = history["amount"].mean()
    else:
        user_avg = current_amount  # Default if new user

    # 4. Predict
    features = [[current_amount, user_avg]]
    prediction = model.predict(features)

    return {
        "user_id": user_id,
        "is_fraud": int(prediction[0]),
        "calculated_avg": user_avg,
    }
