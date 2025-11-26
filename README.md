# ClasificadorIntArt
Sistema de detección de IA usada en trabajos académicos
Descripción General

Este proyecto consiste en un sistema capaz de clasificar textos y determinar si fueron escritos por un estudiante humano o generados por una inteligencia artificial (IA).
Para lograrlo, se entrenó un modelo de aprendizaje estadístico basado en Regresión Logística, utilizando un conjunto masivo de textos etiquetados y procesados mediante extracción de características lingüísticas.

El sistema final está completamente desplegado como una aplicación web, permitiendo al usuario ingresar texto libre y obtener:

La predicción (IA o estudiante)

Las probabilidades asociadas a cada clase

Una interfaz sencilla y amigable

Características del Sistema

El proyecto implementa:

✔ Extracción automática de características lingüísticas, como:

Longitud promedio de oraciones

Longitud promedio de palabras

Riqueza léxica (TTR)

Frecuencia de conectores lógicos

Densidad de puntuación

✔ Escalado de variables mediante StandardScaler

✔ Modelo predictivo entrenado con:

Regresión Logística

250,000 textos procesados

División en entrenamiento y prueba

Métricas de desempeño (accuracy, classification report)

✔ API desarrollada con FastAPI

✔ Frontend simple con HTML + CSS + JS

✔ Despliegue en Render

Versión en Línea (Deploy)
La aplicación completa se encuentra desplegada en:
🔗 https://clasificadorintart.onrender.com
Desde esa página puedes escribir cualquier texto y obtener la clasificación en tiempo real.

Entrenamiento del Modelo
El modelo fue entrenado utilizando el archivo:
dataset_features.csv con más de 250,000 textos, procesados mediante la función extract_features().

Se utilizó:
Escalado con StandardScaler
División 80%/20%
Entrenamiento con LogisticRegression(max_iter=200)
Los modelos generados fueron guardados como:
model_logreg.pkl
scaler.pkl

Tecnologías Utilizadas:
Python 3.10+
FastAPI
scikit-learn
NLTK
Uvicorn
HTML + CSS + JavaScript
Render (deploy del backend y frontend)
Git LFS (manejo de archivos grandes)

