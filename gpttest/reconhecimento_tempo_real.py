import cv2
import mediapipe as mp
import joblib

# Carregar modelo treinado
modelo = joblib.load("modelo_libras.pkl")

# Iniciar MediaPipe e webcam
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

print("Reconhecimento em tempo real iniciado. Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            entrada = []
            for ponto in hand_landmarks.landmark:
                entrada.extend([ponto.x, ponto.y, ponto.z])

            letra = modelo.predict([entrada])[0]
            cv2.putText(frame, f"Letra: {letra}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento de Letras em Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()