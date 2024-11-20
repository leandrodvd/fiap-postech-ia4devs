import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import pandas as pd

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

report_has_face = []
report_dominant_emotion = []
report_face_front = []

def generate_report():
    print(len(report_has_face), len(report_dominant_emotion), len(report_face_front))
    report = pd.DataFrame({
        'frame_index': np.arange(len(report_has_face)),
        'has_face': report_has_face,
        'dominant_emotion': report_dominant_emotion,
        'face_front': report_face_front
    })
    report.to_csv('report.csv', index=False)

    total_frames = len(report_has_face)
    total_faces = sum(report_has_face)
    total_faces_front = sum(report_face_front)

    emotions = report['dominant_emotion'].unique()
    count_has_face_not_front = ((report['has_face'] == True) & (report['face_front'] == False)).sum()
    report_txt = "Total de frames analisados: " + str(total_frames) + "\n"
    report_txt += "Total de frames com face detectada: " + str(total_faces) + "\n"
    report_txt += "Total de frames com face de frente: " + str(total_faces_front) + "\n"
    report_txt += "Total de frames com face detectada mas não de frente: " + str(count_has_face_not_front) + "\n"
    report_txt += "Emoções detectadas: " + str(emotions) + "\n"
    # save to a txt file
    with open('report.txt', 'w') as f:
        f.write(report_txt)
    return report

def is_face_front(landmarks):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    return left_eye.x < left_ear.x and right_eye.x > right_ear.x and left_eye.x > right_eye.x

def analyze_pose(frame):
    # Converter o frame para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o frame para detectar a pose
    results = pose.process(rgb_frame)

    face_front = False

    # Desenhar as anotações da pose no frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        face_front = is_face_front(results.pose_landmarks.landmark)
        cv2.putText(frame, "face_front:"+str(face_front), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    report_face_front.append(face_front)

    return frame;

def analyze_face(frame):
    # Analisar o frame para detectar faces e expressões
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    has_face = False
    dominant_emotion = ""

    # print(frame_index)
    # print(result)
    # Iterar sobre cada face detectada
    for face in result:
        if face['face_confidence'] > 0.5:
            has_face = True
            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

            # Obter a emoção dominante
            dominant_emotion = face['dominant_emotion']

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escrever a emoção dominante acima da face
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    report_has_face.append(has_face)
    report_dominant_emotion.append(dominant_emotion)

    return frame

def analyze_video(video_path, output_path):
    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Largura: {width}, Altura: {height}, FPS: {fps}, Total de frames: {total_frames}")
    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop para processar cada frame do vídeo
    frame_index = 0;
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # if (frame_index > 350):
        #     break
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        frame = analyze_face(frame)
        frame = analyze_pose(frame)

        cv2.putText(frame, str(frame_index), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)
        frame_index += 1
    generate_report()
    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'output_video.mp4')  # Nome do vídeo de saída

# Chamar a função para analisar o vídeo e salvar o vídeo processado e relatório da análise
analyze_video(input_video_path, output_video_path)