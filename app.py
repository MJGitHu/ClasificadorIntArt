from flask import Flask, request, jsonify, render_template
import joblib
from features import extract_features_single
import numpy as np

app = Flask(__name__)

# Cargar modelo
model = joblib.load("model_rf.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    # Extraer features
    features = extract_features_single(text)

    # Convertir a array 2D
    features = np.array(features).reshape(1, -1)

    # Predicción
    pred = model.predict(features)[0]

    return jsonify({"prediction": pred})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
