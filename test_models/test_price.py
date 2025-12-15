import pickle
import pandas as pd

# Load the trained model
with open("models/price_model.pkl", "rb") as f:
    price_model = pickle.load(f)

# Example new transaction
sample_transaction = pd.DataFrame({
    "brand": ["Levis"],
    "category": ["Clothing"],
    "material": ["Cotton"],
    "user_rating": [4.0]
})

# One-hot encode features like training
sample_transaction_encoded = pd.get_dummies(sample_transaction)

# Add missing columns with zeros
for col in price_model.feature_names_in_:
    if col not in sample_transaction_encoded.columns:
        sample_transaction_encoded[col] = 0
sample_transaction_encoded = sample_transaction_encoded[price_model.feature_names_in_]

# Predict price
predicted_price = price_model.predict(sample_transaction_encoded)
print("Predicted Price:", predicted_price[0])
