from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nltk
import re
from nltk import word_tokenize, sent_tokenize
import pandas as pd

nltk.download('punkt')

# Cargar modelo y scaler
model = joblib.load("model_logreg.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

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
    pred = model.predict(scaled)[0]

    return {
        "prediction": pred,
        "label": "🤖 IA" if pred == "ai" else "🧑‍🎓 Estudiante"
    }
