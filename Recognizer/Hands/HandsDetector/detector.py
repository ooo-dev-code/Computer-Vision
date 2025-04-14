import cv2 as cv
import mediapipe as mp

video = cv.VideoCapture('video')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

while video.isOpened():
    
    ret, frame = video.read()
    if not ret:
        break
    
    output = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    results = hands.process(output)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * output.shape[1]), int(landmark.y * output.shape[0])
                cv.circle(output, (x, y), 5, (255, 120, 120), -1)
    
    cv.imshow("Video", frame)
    

    if cv.waitKey(5) & 0xFF ==ord('d'):
        break
    
video.release()
cv.destroyAllWindows()
