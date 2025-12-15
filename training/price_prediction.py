import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
transactions = pd.read_excel("data/transactions.xlsx")
X = pd.get_dummies(transactions[["brand", "category", "material", "user_rating"]])
y = transactions["price"]
price_model = LinearRegression()
price_model.fit(X, y)
with open("models/price_model.pkl", "wb") as f:
    pickle.dump(price_model, f)
print("Price prediction model saved successfully")
