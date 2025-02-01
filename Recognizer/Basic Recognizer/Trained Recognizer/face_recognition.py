import cv2 as cv
import numpy as np

haar = cv.CascadeClassifier(r'OpenCV\Face Recognition\HAAR\haar_face.xml')

people = ["Ben Afflek", "Nelson Mandela", "Squezzie"]
# features = np.load("features.npy", allow_pickle=True)
# labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read(r'C:\Users\moham\OneDrive\Bureau\anxio_meter\OpenCV\Face Recognition\OPENCV\face_trained.yml')

img = cv.imread(r'OpenCV\Photos\val\img2.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

faces_rect = haar.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f"This person is {str(people[label])} with a confidence of {confidence}")
    
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    
cv.imshow("Img", img)

cv.waitKey(0)
