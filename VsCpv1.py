import cv2
import mediapipe as mp

# Inicia o detector de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Função auxiliar para saber se um dedo está estendido
def dedo_levantado(pontos, dedo):
    if dedo == "polegar":
        return pontos[4].x < pontos[3].x  # Pode inverter dependendo da mão
    elif dedo == "indicador":
        return pontos[8].y < pontos[6].y
    elif dedo == "medio":
        return pontos[12].y < pontos[10].y
    elif dedo == "anelar":
        return pontos[16].y < pontos[14].y
    elif dedo == "mindinho":
        return pontos[20].y < pontos[18].y
    return False

# Função para detectar apenas as vogais da Libras
def detectar_vogal(pontos):
    dedos = {
        "polegar": dedo_levantado(pontos, "polegar"),
        "indicador": dedo_levantado(pontos, "indicador"),
        "medio": dedo_levantado(pontos, "medio"),
        "anelar": dedo_levantado(pontos, "anelar"),
        "mindinho": dedo_levantado(pontos, "mindinho"),
    }

    if all(not v for v in dedos.values()):
        return "Letra A"
    elif all(v for v in dedos.values()):
        return "Letra E"
    elif dedos == {"polegar": False, "indicador": False, "medio": False, "anelar": False, "mindinho": True}:
        return "Letra I"
    elif dedos == {"polegar": True, "indicador": True, "medio": True, "anelar": True, "mindinho": False}:
        return "Letra O"
    elif dedos == {"polegar": False, "indicador": False, "medio": True, "anelar": False, "mindinho": True}:
        return "Letra U"
    else:
        return "Gesto desconhecido 🤔"

# Iniciar a captura da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imagem_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            letra = detectar_vogal(hand_landmarks.landmark)
            cv2.putText(frame, letra, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imshow("Reconhecimento de Vogais em Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
