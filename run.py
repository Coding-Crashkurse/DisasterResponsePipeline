from app import app
import pickle
from app import tokenize
import joblib


if __name__ == "__main__":
    model = joblib.load("models/classifier.pkl")
    app.run()

