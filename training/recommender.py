import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

ratings = ratings = pd.read_excel("data/ratings.xlsx")

user_item = ratings.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating"
).fillna(0)

similarity = cosine_similarity(user_item)

with open("models/recommender.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("Recommender model saved successfully")
import pickle
import pandas as pd

# Load recommender
with open("models/recommender.pkl", "rb") as f:
    similarity = pickle.load(f)

print(similarity)