import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

label = "i"  # Mude para a letra desejada antes de coletar
arquivo_csv = open("gestos.csv", mode='a', newline='')
writer = csv.writer(arquivo_csv)

print("Pressione 's' para salvar o gesto atual. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                dados = [label]
                for ponto in hand_landmarks.landmark:
                    dados.extend([ponto.x, ponto.y, ponto.z])
                writer.writerow(dados)
                print(f"Gesto da letra '{label}' salvo!")

    cv2.putText(frame, f"Coletando letra: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Coletor de Gestos - Libras", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
arquivo_csv.close()
cv2.destroyAllWindows()
