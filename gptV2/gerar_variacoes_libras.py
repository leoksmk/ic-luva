import cv2
import mediapipe as mp
import joblib
import imageio
import numpy as np
import os
from imgaug import augmenters as iaa

modelo = joblib.load("modelo_libras.pkl")
os.makedirs("imagens_geradas", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
desenhador = mp.solutions.drawing_utils

aumentador = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Multiply((0.8, 1.2)),
    iaa.AdditiveGaussianNoise(scale=0.02*255),
    iaa.Affine(rotate=(-20, 20), scale=(0.9, 1.1)),
    iaa.ChangeColorTemperature((4000, 10000))
])

cap = cv2.VideoCapture(0)
contador = 0
print("Pressione ESPAÇO para capturar a mão. ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            desenhador.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            entrada = [p.x for p in hand_landmarks.landmark] + \
                      [p.y for p in hand_landmarks.landmark] + \
                      [p.z for p in hand_landmarks.landmark]
            letra = modelo.predict([entrada])[0]
            cv2.putText(frame, f"Letra: {letra}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Captura para Geração de Dados", frame)
    tecla = cv2.waitKey(1)

    if tecla == 27:
        break
    elif tecla == 32 and resultado.multi_hand_landmarks:
        nome_base = f"letra_{letra}_{contador}"
        caminho_original = f"imagens_geradas/{nome_base}_original.jpg"
        cv2.imwrite(caminho_original, frame)

        imagem = imageio.imread(caminho_original)
        imagens_aumentadas = aumentador(images=[imagem]*30)

        for i, img in enumerate(imagens_aumentadas):
            caminho = f"imagens_geradas/{nome_base}_aug_{i}.jpg"
            imageio.imwrite(caminho, img)

        print(f"30 variações da letra '{letra}' salvas em 'imagens_geradas/'")
        contador += 1

cap.release()
cv2.destroyAllWindows()
