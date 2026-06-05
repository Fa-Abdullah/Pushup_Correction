import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

# Load the trained model
with open('pushup_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

queue_size = 5
pred_queue = deque(maxlen=queue_size)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    landmarks_list = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks_list += [lm.x, lm.y, lm.z, lm.visibility]
        pred = model.predict([landmarks_list])[0]
        pred = "Correct" if pred == 1 else "Wrong"
    else:
        pred = "No person detected"

    pred_queue.append(pred)
    if len(pred_queue) == queue_size:
        smoothed_pred = max(set(pred_queue), key=pred_queue.count)
    else:
        smoothed_pred = pred

    color = (0, 255, 0) if smoothed_pred == "Correct" else (0, 0, 255)
    cv2.putText(frame, f'Push-up: {smoothed_pred}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("AI Pushup Correction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
