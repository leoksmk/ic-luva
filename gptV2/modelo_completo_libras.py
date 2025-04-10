import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
from skimage.io import imread
from skimage.transform import resize

PASTA_IMAGENS = "imagens_geradas"
TAMANHO = (128, 128)

print("Coletando imagens...")
arquivos = [f for f in os.listdir(PASTA_IMAGENS) if f.endswith(".jpg")]
X = []
y = []

for nome in arquivos:
    caminho = os.path.join(PASTA_IMAGENS, nome)
    imagem = imread(caminho, as_gray=True)
    imagem = resize(imagem, TAMANHO).flatten()
    X.append(imagem)
    partes = nome.split("_")
    letra = partes[1] if len(partes) > 1 else "?"
    y.append(letra)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Treinando modelo com imagens aumentadas...")
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

print("Avaliando o modelo...")
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(modelo, "modelo_libras_imagem.pkl")
print("Modelo salvo como 'modelo_libras_imagem.pkl'")
