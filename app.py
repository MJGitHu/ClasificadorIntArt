from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import nltk
import re
from nltk import word_tokenize, sent_tokenize
import pandas as pd

nltk.download("punkt")

# Cargar modelo y scaler
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

def extract_features(text):
    try:
        tokens = word_tokenize(text.lower())
        words = [w for w in tokens if w.isalpha()]
        sentences = sent_tokenize(text)

        if len(words) == 0 or len(sentences) == 0:
            return [0]*5

        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(w) for w in words) / len(words)
        ttr = len(set(words)) / len(words)

        connectors = [
            "however", "therefore", "moreover", "furthermore", "thus",
            "in conclusion", "for example", "on the other hand"
        ]
        connective_ratio = sum(text.lower().count(c) for c in connectors) / len(sentences)
        punctuation_density = len(re.findall(r"[,.!?;:]", text)) / len(words)

        return [
            avg_sentence_length,
            avg_word_length,
            ttr,
            connective_ratio,
            punctuation_density
        ]
    except:
        return [0]*5


@app.post("/predict")
def predict(data: InputText):
    features = extract_features(data.text)

    df = pd.DataFrame([features], columns=[
        "avg_sentence_length",
        "avg_word_length",
        "ttr",
        "connective_ratio",
        "punctuation_density"
    ])

    scaled = scaler.transform(df)

    # Predicción dura
    pred = model.predict(scaled)[0]

    # Probabilidades
    probs = model.predict_proba(scaled)[0]
    prob_student = float(probs[model.classes_.tolist().index("student")])
    prob_ai = float(probs[model.classes_.tolist().index("ai")])

    return {
        "prediction": pred,
        "label": "🤖 IA" if pred == "ai" else "🧑‍🎓 Estudiante",
        "probabilities": {
            "student": round(prob_student * 100, 2),
            "ai": round(prob_ai * 100, 2)
        }
    }
