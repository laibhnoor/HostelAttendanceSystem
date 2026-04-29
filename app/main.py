import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cascade_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise Exception("Haar cascade not loaded! Check file path.")

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()