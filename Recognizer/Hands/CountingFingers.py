import cv2 as cv
import numpy as np
import mediapipe as mp
import random

# Initialize the webcam
cap = cv.VideoCapture(0)

# Set the frame rate
fps = 320
cap.set(cv.CAP_PROP_FPS, fps)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
index = random.randint(0, 3)

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            finger_dic = {}
            
            scissors_fingers = 0
            standing_fingers = 0
            down_fingers = 0
            choice = ""
            ok = False;
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i in [8, 12]:
                    scissors_fingers+=1
                if i in [8, 12, 16, 20]: 
                    if landmark.y < hand_landmarks.landmark[i - 3].y:  
                        standing_fingers += 1
                    else:
                        down_fingers += 1
                if i == 4:  # Tip of Thumb
                    if landmark.y < hand_landmarks.landmark[12].y:
                        ok = True
                    if landmark.y < hand_landmarks.landmark[5].y:  
                        standing_fingers += 1
                    else:
                        down_fingers += 1
            cv.putText(frame, f'Standing Fingers: {standing_fingers}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f'Down Fingers: {down_fingers}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                                
            choices = ["Rock", "Paper", "Scissors"]
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i == 9:
                    finger_dic["TopHand"] = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if i % 4 == 0:
                    finger_name = finger_names[(i // 4)-1]
                    finger_dic[finger_name] = [int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv.putText(frame, finger_name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)
                    cv.circle(frame, (x, y), 5, (255, 120, 120), -1)

    cv.imshow('Hand Tracking', frame)

    if cv.waitKey(5) & 0xFF ==ord('d'):
        break

cap.release()
cv.destroyAllWindows()
