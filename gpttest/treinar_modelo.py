import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

# Carregar os dados
print("Carregando dados...")
dados = pd.read_csv("gestos.csv", header=None)
X = dados.iloc[:, 1:64]  # 21 pontos * 3 coordenadas = 63 colunas
y = dados.iloc[:, 0]     # Primeira coluna é o label (letra)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelo
print("Treinando modelo KNN...")
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# Avaliar
print("Avaliando modelo...")
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# Salvar modelo
joblib.dump(modelo, "modelo_libras.pkl")
print("Modelo salvo como 'modelo_libras.pkl'")
