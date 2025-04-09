import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def dedo_levantado(pontos, dedo):
    if dedo == "polegar":
        return pontos[4].x < pontos[3].x
    elif dedo == "indicador":
        return pontos[8].y < pontos[6].y
    elif dedo == "medio":
        return pontos[12].y < pontos[10].y
    elif dedo == "anelar":
        return pontos[16].y < pontos[14].y
    elif dedo == "mindinho":
        return pontos[20].y < pontos[18].y
    return False

def detectar_letra(pontos):
    dedos = {
        "polegar": dedo_levantado(pontos, "polegar"),
        "indicador": dedo_levantado(pontos, "indicador"),
        "medio": dedo_levantado(pontos, "medio"),
        "anelar": dedo_levantado(pontos, "anelar"),
        "mindinho": dedo_levantado(pontos, "mindinho"),
    }

    if all(not v for v in dedos.values()):
        return "A"
    elif dedos == {"polegar": False, "indicador": True, "medio": True, "anelar": True, "mindinho": True}:
        return "B"
    elif all(dedos.values()):  # Todos levantados (forma de "C" depende da curvatura)
        return "C (estimativa)"
    elif dedos == {"polegar": False, "indicador": True, "medio": False, "anelar": False, "mindinho": False}:
        return "D"
    elif all(v for v in dedos.values()):
        return "E"
    elif dedos["polegar"] and dedos["indicador"]:  # Faria um círculo, difícil sem geometria
        return "F (estimativa)"
    elif dedos["polegar"] and dedos["indicador"]:
        return "G (estimativa)"
    elif dedos["indicador"] and dedos["medio"]:
        return "H (estimativa)"
    elif dedos == {"polegar": False, "indicador": False, "medio": False, "anelar": False, "mindinho": True}:
        return "I"
    elif dedos == {"polegar": False, "indicador": False, "medio": False, "anelar": False, "mindinho": True}:
        return "J (estimativa)"
    elif dedos["polegar"] and dedos["indicador"] and dedos["medio"]:
        return "K (estimativa)"
    elif dedos == {"polegar": True, "indicador": True, "medio": False, "anelar": False, "mindinho": False}:
        return "L"
    elif not dedos["polegar"] and not dedos["indicador"] and not dedos["medio"]:
        return "M (estimativa)"
    elif not dedos["polegar"] and not dedos["indicador"]:
        return "N (estimativa)"
    elif all(dedos.values()):
        return "O"
    elif dedos["polegar"] and dedos["indicador"] and dedos["medio"]:
        return "P (estimativa)"
    elif dedos["polegar"] and dedos["indicador"]:
        return "Q (estimativa)"
    elif dedos["indicador"] and dedos["medio"]:
        return "R (estimativa)"
    elif all(not v for v in dedos.values()):
        return "S"
    elif dedos["polegar"] and not dedos["indicador"] and not dedos["medio"]:
        return "T (estimativa)"
    elif dedos["indicador"] and dedos["medio"] and not dedos["anelar"]:
        return "U"
    elif dedos["indicador"] and dedos["medio"] and not dedos["anelar"] and not dedos["mindinho"]:
        return "V"
    elif dedos["indicador"] and dedos["medio"] and dedos["anelar"]:
        return "W"
    elif dedos["indicador"]:
        return "X (estimativa)"
    elif dedos["polegar"] and dedos["mindinho"]:
        return "Y"
    elif dedos["indicador"]:
        return "Z (estimativa)"
    else:
        return "Desconhecido"

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

            letra = detectar_letra(hand_landmarks.landmark)
            cv2.putText(frame, f"Letra: {letra}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Alfabeto em Libras (Básico)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
