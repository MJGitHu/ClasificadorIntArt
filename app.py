from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import nltk
import re
from nltk import word_tokenize, sent_tokenize
import pandas as pd

#Recursos para trabajar el texto.
nltk.download("punkt")
nltk.download('punkt_tab')  
nltk.download('averaged_perceptron_tagger')

# Cargar modelo y scaler (regresión logística)
model = joblib.load("model_logreg.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# ============
#  SERVIR WEB
# ============

# Servir carpeta "static"
app.mount("/static", StaticFiles(directory="static"), name="static")

# Servir index.html al entrar a la página
@app.get("/")
def read_root():
    return FileResponse("templates/index.html")

# ============
#  API DE ML
# ============

class InputText(BaseModel):
    text: str

#tokenizador
def extract_features(text):
    text = text.strip()
    if len(text) < 5:
        return [0]*5

    try:
        sentences = sent_tokenize(text)
        tokens = word_tokenize(text.lower())
        words = [w for w in tokens if w.isalpha()]

        if len(words) < 3 or len(sentences) < 1:
            return [0]*5

        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_word_length = sum(len(w) for w in words) / len(words)
        ttr = len(set(words)) / len(words)

        connectors = [
            "however", "therefore", "moreover", "furthermore", "thus",
            "in conclusion", "for example", "on the other hand"
        ]
        connective_ratio = sum(text.lower().count(c) for c in connectors) / max(1, len(sentences))

        punctuation_density = len(re.findall(r"[,.!?;:]", text)) / len(words)

        return [
            avg_sentence_length,
            avg_word_length,
            ttr,
            connective_ratio,
            punctuation_density
        ]

    except Exception as e:
        print("FEATURE ERROR:", e)
        return [0]*5


@app.post("/predict")
def predict(data: InputText):
    text = data.text

    # Extraer características
    features = extract_features(text)
    print("FEATURES:", features)  # <--- Debug
    
    df = pd.DataFrame([features], columns=[
        "avg_sentence_length",
        "avg_word_length",
        "ttr",
        "connective_ratio",
        "punctuation_density"
    ])

    # Escalar
    scaled = scaler.transform(df)
    print("SCALED:", scaled)  # <--- Debug

    # Predicción y probabilidades
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]

    probs_dict = dict(zip(model.classes_, probs))

    print("PRED:", pred)
    print("PROBS:", probs_dict)

    return {
    "prediction": pred,
    "probabilities": {
        "student": round(probs_dict["student"] * 100, 2),
        "ai": round(probs_dict["ai"] * 100, 2)
    }
}

