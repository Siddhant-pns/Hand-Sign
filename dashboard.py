import tkinter as tk
from tkinter import Label, Button, Entry, Canvas
import cv2
import mediapipe as mp
import numpy as np
from sign_model import SignModel
from text_to_speech import speak
import threading
from PIL import Image, ImageTk

# Initialize Tkinter GUI (Root Window First)
root = tk.Tk()
root.title("Hand Sign Translator")
root.geometry("900x500")

# Global Variables (Define After Root Creation)
mode = tk.StringVar(root)  # Bind to root
mode.set("Select Mode")
current_sign = tk.StringVar(root)
current_features = None

# Initialize Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load Sign Model
sign_model = SignModel()

# Function to Extract Features from Hand Landmarks
def extract_features(hand_landmarks):
    """ Extracts 21 (x, y) coordinates from hand landmarks and normalizes them """
    features = []
    base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y  # Wrist Position (Base Point)
    
    for lm in hand_landmarks.landmark:
        norm_x = lm.x - base_x  # Normalize X relative to wrist
        norm_y = lm.y - base_y  # Normalize Y relative to wrist
        features.append(norm_x)
        features.append(norm_y)
    
    return np.array(features)


# Function to Process Camera Feed Inside UI
subtitle_text = ""  # Store continuous subtitles
last_detected_sign = None  # Store last detected sign
hand_visible = False  # Track if a hand is visible

def update_camera():
    global current_features, subtitle_text, last_detected_sign, hand_visible
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_sign = "Unknown"
        current_features = None
        hand_present = False  # Flag to track hand presence

        if results.multi_hand_landmarks:
            hand_present = True  # Hand is detected
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_features = extract_features(hand_landmarks)

                if current_features is not None:
                    detected_sign = sign_model.predict(current_features)

                    # **TEST MODE: Update subtitle only when a new sign appears**
                    if mode.get() == "Test Mode":
                        if detected_sign != "Unknown":
                            if detected_sign != last_detected_sign or last_detected_sign == "Unknown":
                                subtitle_text += " " + detected_sign
                                last_detected_sign = detected_sign  # Update last detected sign

                    # **TRAIN MODE: Show detected sign below input field**
                    if mode.get() == "Train Mode":
                        detected_sign_label.config(text=f"Detected: {detected_sign}")

        # If no hands are detected, reset tracking for re-appearance
        if not hand_present:
            hand_visible = False  # Hand is not visible
            last_detected_sign = "Unknown"  # Reset last detected sign
        else:
            hand_visible = True  # Hand is visible

        # Convert Frame for Tkinter Canvas
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        # Update Subtitle Text
        subtitle_label.config(text=subtitle_text.strip())

        root.update_idletasks()
        root.after(5)

    cap.release()


# Function to Start Camera Thread
def start_camera():
    threading.Thread(target=update_camera, daemon=True).start()

# Function to Set Train Mode
def set_train_mode():
    mode.set("Train Mode")
    current_sign.set("Show a sign & Enter name below")

# Function to Set Test Mode
def set_test_mode():
    mode.set("Test Mode")
    current_sign.set("Waiting for Detection...")

# Function to Save Sign
def save_sign():
    global current_features
    if mode.get() == "Train Mode":
        sign_name = sign_entry.get().strip()
        if sign_name and current_features is not None:
            sign_model.add_new_sign(current_features, sign_name)
            sign_entry.delete(0, tk.END)  # Clear Input Box
            current_sign.set(f"✅ '{sign_name}' Saved Successfully!")
        else:
            current_sign.set("⚠ No Hand Detected! Try Again.")


# Fixed UI Layout
left_frame = tk.Frame(root, width=300, height=500, bg="lightgray")
left_frame.pack(side="left", fill="both", expand=False)

right_frame = tk.Frame(root, width=600, height=500)
right_frame.pack(side="right", fill="both", expand=True)
# Label to show detected sign in Train Mode
detected_sign_label = Label(left_frame, text="Detected: ", font=("Arial", 12), fg="black", bg="lightgray")
detected_sign_label.pack(pady=10)


# Left Side UI (Controls)
Label(left_frame, text="Hand Sign Translator", font=("Arial", 16), bg="lightgray").pack(pady=20)
Button(left_frame, text="Train Mode", command=set_train_mode, font=("Arial", 12), bg="lightgreen").pack(pady=10)
Button(left_frame, text="Test Mode", command=set_test_mode, font=("Arial", 12), bg="lightblue").pack(pady=10)
Label(left_frame, text="Enter Sign Name:", font=("Arial", 12), bg="lightgray").pack(pady=10)

sign_entry = Entry(left_frame, font=("Arial", 12))
sign_entry.pack(pady=5)

Button(left_frame, text="Save Sign", command=save_sign, font=("Arial", 12), bg="orange").pack(pady=10)
Label(left_frame, textvariable=current_sign, font=("Arial", 14), fg="blue", bg="lightgray").pack(pady=20)
Button(left_frame, text="Exit", command=root.quit, font=("Arial", 12), bg="red").pack(pady=10)

# Right Side UI (Fixed Camera Panel)
camera_panel = tk.Frame(right_frame, width=500, height=350)  # 2/3rd of right section
camera_panel.pack(side="top", pady=10)
camera_label = Label(camera_panel)
camera_label.pack()

# Subtitles Below Camera (Bold Text)
subtitle_label = Label(right_frame, text="", font=("Arial", 16, "bold"), fg="blue")
subtitle_label.pack(side="bottom", pady=20)



# Start Camera Thread
start_camera()

root.mainloop()
