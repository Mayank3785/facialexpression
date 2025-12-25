import cv2
import numpy as np
from keras.models import load_model

model = load_model("model/emotion_model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi / 255.0
        roi = np.reshape(roi, (1,48,48,1))

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
