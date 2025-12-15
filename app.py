from fastapi import FastAPI
import pickle
import logging


logging.basicConfig(filename='logs/app.log', level=logging.INFO)


app = FastAPI()


price_model = pickle.load(open('models/price_model.pkl', 'rb'))
category_model, vectorizer = pickle.load(open('models/category_model.pkl', 'rb'))


@app.get('/')
def home():
return {'status': 'BabbaFly ML API Running'}


@app.post('/predict-price')
def predict_price(user_rating: float):
try:
price = price_model.predict([[user_rating]])
logging.info('Price predicted')
return {'predicted_price': float(price[0])}
except Exception as e:
logging.error(str(e))
return {'error': 'Prediction failed'}


@app.post('/predict-category')
def predict_category(text: str):
vec = vectorizer.transform([text])
cat = category_model.predict(vec)
return {'category': cat[0]}