from flask import Flask, request, jsonify
from joblib import load
from features import extract_features_single
import numpy as np

app = Flask(__name__)

model = load("model_rf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    features = extract_features_single(text)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": prediction
    })

@app.route("/", methods=["GET"])
def home():
    return "API funcionando."

if __name__ == "__main__":
    app.run()
