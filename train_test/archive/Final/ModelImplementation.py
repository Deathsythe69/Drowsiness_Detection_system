import cv2
import tensorflow as tf
# import tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = tf.keras.models.load_model(r'Models/model_v2.h5')

thicc = 2

# lbl=['Close','Open']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
path = os.getcwd()
mixer.init()
sound = mixer.Sound(r'alarm.wav')
cap = cv2.VideoCapture(0)
Score = 0
while True:
    ret, frame = cap.read()
    height, width = frame.shape[0:2]

    #     print(frame.shape)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 1)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (ex, ey, ew, eh) in eyes:
        #         cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        eye = frame[ey:ey + eh, ex:ex + ew]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        #         print(prediction)

        if prediction[0][0] > 0.50:
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (5, 30, 252), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score:" + str(Score), (100, height - 20), font, 1, (252, 5, 5), 1, cv2.LINE_AA)
            Score = Score + 1
            if (Score > 5):
                cv2.imwrite(os.path.join(path, 'Images/status.jpg'), frame)
                try:
                    sound.play()
                except:
                    pass
                if (thicc < 16):
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if (thicc < 2):
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        elif prediction[0][1] > 0.50:

            cv2.putText(frame, "Open", (10, height - 20), font, 1, (5, 30, 252), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score:" + str(Score), (100, height - 20), font, 1, (252, 5, 5), 1, cv2.LINE_AA)
            Score = Score - 1
            if (Score < 0):
                Score = 0

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()