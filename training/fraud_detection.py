import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
fraud = pd.read_excel("data/fraud_listings.xlsx")
X = fraud[["price", "description_length", "images_count"]]
model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X)
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Fraud detection model saved successfully")
with open("models/fraud_model.pkl", "rb") as f:
    fraud_model = pickle.load(f)
sample_listing = pd.DataFrame({
    "price": [1200],
    "description_length": [35],
    "images_count": [2]
})
fraud_pred = fraud_model.predict(sample_listing)
fraud_label = "Fraud" if fraud_pred[0] == -1 else "Not Fraud"
print(fraud_label)  
