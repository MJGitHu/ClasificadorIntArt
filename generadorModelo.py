import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Cargar dataset
df = pd.read_csv("dataset_features.csv")

X = df.drop("generated", axis=1)
y = df["generated"]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar modelo
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Evaluación
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar modelo y scaler
joblib.dump(log_reg, "model_logreg.pkl")
joblib.dump(scaler, "scaler.pkl")

files.download("model_logreg.pkl")
files.download("scaler.pkl")
print("Modelo guardado correctamente.")
