# train.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")

# MANUAL FEATURE ENGINEERING:
# Calculate average spend per user up to that row
df["avg_spend"] = df.groupby("user_id")["amount"].transform(
    lambda x: x.expanding().mean()
)

# Train model
X = df[["amount", "avg_spend"]]
y = df["is_fraud"]
model = RandomForestClassifier().fit(X, y)

# Save model
joblib.dump(model, "fraud_model.pkl")
print("Model trained and saved as fraud_model.pkl")
