import nltk
from nltk import word_tokenize, sent_tokenize
import re

nltk.download("punkt")

def extract_features_single(text):
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha()]
    sentences = sent_tokenize(text)

    if len(words) == 0 or len(sentences) == 0:
        return [0, 0, 0, 0, 0]

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
