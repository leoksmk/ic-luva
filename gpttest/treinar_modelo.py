import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

print("Carregando dados...")
dados = pd.read_csv("gestos.csv", header=None)
X = dados.iloc[:, 1:64]
y = dados.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Treinando modelo KNN...")
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

print("Avaliando modelo...")
print(classification_report(y_test, modelo.predict(X_test)))

joblib.dump(modelo, "modelo_libras.pkl")
print("Modelo salvo como 'modelo_libras.pkl'")
