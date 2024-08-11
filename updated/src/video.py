import numpy as np
import cv2
import pickle

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_smile.xml')

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("src/recognizers/face-trainner.yml")

# Load the labels
labels = {"person_name": 1}
with open("src/pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# Replace this with the path to your video file
cap = cv2.VideoCapture('D:/VisualStudioCode/HIR/scenes2022/Calloused_Hands/Calloused_Hands-1.webm')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels.get(id_, "Unknown")
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)
        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
