import cv2
import mediapipe as mp
import numpy as np
from sign_model import SignModel  # ML Model
from text_to_speech import speak  # Text-to-Speech

# Initialize Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load Sign Language Model
sign_model = SignModel()

# Start Webcam
cap = cv2.VideoCapture(0)

def extract_features(hand_landmarks):
    """ Extracts 21 (x, y) coordinates from hand landmarks """
    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_sign = "Unknown"
    features = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_features(hand_landmarks)

            # Predict Sign if Features Exist
            if features is not None:
                detected_sign = sign_model.predict(features)

    # Show Result on Screen
    cv2.putText(frame, f"Sign: {detected_sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Camera Feed
    cv2.imshow("Hand Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('n') and features is not None:  # Add New Sign
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()  # Hide the root window
        new_label = simpledialog.askstring("New Sign", "Enter sign label:")

        if new_label:
            sign_model.add_new_sign(features, new_label)
            print(f"✅ New sign '{new_label}' added successfully!")
    
    elif key == ord('s'):  # Speak Detected Sign
        speak(detected_sign)

cap.release()
cv2.destroyAllWindows()
