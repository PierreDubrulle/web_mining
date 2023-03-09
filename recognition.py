import cv2
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import mediapipe as mp
import tensorflow as tf


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer.yml')

names = ['antoine', 'axel','christelle_cor', 'christelle_kie', 'chrystelle','fatimetou',
            'florian','hugo','ibtissam','laura','leo','loic','louison','martin','matthieu',
            'pauline','pierre','robin','samuel','tho','titouan']
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

while(True):

    ret, frame = vid.read()

    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)

    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        id, confidence = recognizer.predict(gray_fr[y:y+h,x:x+w])
            
        if (confidence < 100):
            try:
                nom = names[id-1]
                confidence = "  {0}%".format(round(100 - confidence))
            except:
                pass
        else:
            try:
                confidence = "  {0}%".format(round(100 - confidence))
            except:
                pass
        
        cv2.putText(
                frame, 
                str(nom), 
                (x+5,y-5), 
                cv2.FONT_HERSHEY_DUPLEX, 
                1, 
                (255,255,255), 
                2
            )

        cv2.putText(
                frame, 
                str(confidence), 
                (x+5,y+h-5), 
                cv2.FONT_HERSHEY_DUPLEX, 
                1, 
                (255,255,0), 
                1
                )
    cv2.imshow('test', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
vid.relead()
cv2.destroyAllWindows()