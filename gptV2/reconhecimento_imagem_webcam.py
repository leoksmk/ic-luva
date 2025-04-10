import cv2
import joblib
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

modelo = joblib.load("modelo_libras_imagem.pkl")
TAMANHO = (128, 128)

cap = cv2.VideoCapture(0)
print("Pressione ESPAÇO para prever a letra. Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Reconhecimento por Imagem", frame)
    tecla = cv2.waitKey(1)

    if tecla == 27:
        break
    elif tecla == 32:
        imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem = rgb2gray(imagem)
        imagem = resize(imagem, TAMANHO)
        imagem = imagem.flatten().reshape(1, -1)

        letra = modelo.predict(imagem)[0]
        print(f"Letra reconhecida: {letra}")
        cv2.putText(frame, f"Letra: {letra}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Reconhecimento por Imagem", frame)
        cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
