import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

# READ EXCEL FILE (NOT CSV)
fraud = pd.read_excel("data/fraud_listings.xlsx")

X = fraud[["price", "description_length", "images_count"]]

model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X)

with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Fraud detection model saved successfully")
import pickle
import pandas as pd

# Load the trained fraud detection model
with open("models/fraud_model.pkl", "rb") as f:
    fraud_model = pickle.load(f)

# Example listing features
sample_listing = pd.DataFrame({
    "price": [1200],
    "description_length": [35],
    "images_count": [2]
})

# Predict fraud
fraud_pred = fraud_model.predict(sample_listing)
# IsolationForest: -1 = fraud, 1 = not fraud
fraud_label = "Fraud" if fraud_pred[0] == -1 else "Not Fraud"

print(fraud_label)  # Output: 'Not Fraud' (or 'Fraud' depending on model)
