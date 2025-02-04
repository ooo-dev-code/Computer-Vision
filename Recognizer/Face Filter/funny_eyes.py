
import cv2
# Load the pre-trained Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start the video capture from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    eyes = face_cascade.detectMultiScale(gray, scaleFactor=3.1, minNeighbors=5, minSize=(30, 30))
    # Draw circles at the center of the eyes
    for (x, y, w, h) in eyes:
        center_x = x + w // 2
        center_y = y + h // 2
        radius = min(w, h) // 2
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), radius-5, (255, 255, 255), -1)
        cv2.circle(frame, (center_x, center_y), radius-10, (255, 50, 50), -1)
        cv2.circle(frame, (center_x, center_y), radius-20, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), radius-25, (0, 0, 0), -1)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
