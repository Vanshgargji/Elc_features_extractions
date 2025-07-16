import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from math import dist

# Facial landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_EAR(landmarks, img_w, img_h):
    def eye_aspect(eye):
        p1 = np.array([landmarks[eye[1]].x * img_w, landmarks[eye[1]].y * img_h])
        p2 = np.array([landmarks[eye[5]].x * img_w, landmarks[eye[5]].y * img_h])
        p3 = np.array([landmarks[eye[2]].x * img_w, landmarks[eye[2]].y * img_h])
        p4 = np.array([landmarks[eye[4]].x * img_w, landmarks[eye[4]].y * img_h])
        p0 = np.array([landmarks[eye[0]].x * img_w, landmarks[eye[0]].y * img_h])
        p5 = np.array([landmarks[eye[3]].x * img_w, landmarks[eye[3]].y * img_h])
        return (dist(p1, p2) + dist(p3, p4)) / (2.0 * dist(p0, p5))

    left_EAR = eye_aspect(LEFT_EYE)
    right_EAR = eye_aspect(RIGHT_EYE)
    return (left_EAR + right_EAR) / 2.0

def calculate_MAR(landmarks, img_w, img_h):
    p1 = np.array([landmarks[78].x * img_w, landmarks[78].y * img_h])
    p2 = np.array([landmarks[308].x * img_w, landmarks[308].y * img_h])
    p3 = np.array([landmarks[95].x * img_w, landmarks[95].y * img_h])
    p4 = np.array([landmarks[324].x * img_w, landmarks[324].y * img_h])
    vertical1 = dist(p1, p2)
    vertical2 = dist(p3, p4)
    horizontal = dist([landmarks[61].x * img_w, landmarks[61].y * img_h],
                      [landmarks[291].x * img_w, landmarks[291].y * img_h])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Set paths
DATASET_DIR = r"D:\elc_2nd"
OUTPUT_CSV = r"D:\elc_2nd\driver_drowsiness_features.csv"

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Write CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = [f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)] + ['EAR', 'MAR', 'label']
    writer.writerow(header)

    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Cannot load image: {img_path}")
                continue

            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                try:
                    landmarks = result.multi_face_landmarks[0].landmark
                    x_coords = [lm.x for lm in landmarks]
                    y_coords = [lm.y for lm in landmarks]
                    EAR = calculate_EAR(landmarks, w, h)
                    MAR = calculate_MAR(landmarks, w, h)
                    row = x_coords + y_coords + [EAR, MAR, label]
                    writer.writerow(row)
                    print(f"✅ Processed: {img_file} ({label})")
                except Exception as e:
                    print(f"❌ Error in EAR/MAR: {img_file} – {e}")
            else:
                print(f"❌ No face detected in: {img_file}")
