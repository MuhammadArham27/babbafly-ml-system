import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# READ EXCEL FILE (NOT CSV)
products = pd.read_excel("data/products.xlsx")

X = products["title"] + " " + products["description"]
y = products["category"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

with open("models/category_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Category classifier model saved successfully")
with open("models/category_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

sample_text = ["Soft cotton shirt for men"]
sample_vec = vectorizer.transform(sample_text)
prediction = model.predict(sample_vec)
print(prediction)  # Output: ['Clothing']
