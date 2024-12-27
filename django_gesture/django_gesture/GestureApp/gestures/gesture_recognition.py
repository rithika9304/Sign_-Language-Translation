import cv2
import mediapipe as mp
import threading

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global flag to control the video capture
running = False

# Function to check if a finger is up
def is_finger_up(landmarks, finger_tip_idx, finger_dip_idx):
    return landmarks[finger_tip_idx].y < landmarks[finger_dip_idx].y

# Function to detect specific gestures
def detect_gesture(landmarks):
    thumb_up = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x
    thumb_down = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
    index_up = is_finger_up(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP)
    middle_up = is_finger_up(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP)
    ring_up = is_finger_up(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP)
    pinky_up = is_finger_up(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)

    # Detect gestures based on finger positions
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Up"
    elif thumb_down and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Down"
    elif index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        return "Peace (V Sign)"
    elif thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "Call Me"
    elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Fist"
    elif thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "Okay"
    elif index_up and middle_up and ring_up and not pinky_up:
        return "Three Fingers"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "Two Fingers"
    elif index_up and not middle_up and not ring_up and not pinky_up:
        return "One Finger"
    elif thumb_up and index_up and pinky_up and not middle_up and not ring_up:
        return "Rock On"
    elif thumb_up and index_up and middle_up and pinky_up and not ring_up:
        return "Live Long (Vulcan Salute)"
    elif thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "Five Fingers (Open Hand - Stop/Smile)"
    elif index_up and middle_up and ring_up and pinky_up and not thumb_up:
        return "Four Fingers"
    else:
        return "Unknown"

# Function to start video capture and detect gestures
def run_gesture_recognition():
    global running
    cap = cv2.VideoCapture(0)
    running = True

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start and Stop functions for gesture recognition
def start_recognition():
    threading.Thread(target=run_gesture_recognition).start()

def stop_recognition():
    global running
    running = False
