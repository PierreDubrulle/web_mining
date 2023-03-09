import cv2
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import mediapipe as mp
import tensorflow as tf


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer.yml')
ageproto = './models2/age_deploy.prototxt'
agemodel = './models2/age_net.caffemodel'

agenet = cv2.dnn.readNet(ageproto, agemodel)


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

names = ['antoine', 'axel','christelle_cor', 'christelle_kie', 'chrystelle','fatimetou',
            'florian','hugo','ibtissam','laura','leo','loic','louison','martin','matthieu',
            'pauline','pierre','robin','samuel','tho','titouan']
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getFaceBox(net, frame, conf_treshold=0.7):
    frameOpenCvDnn = frame.copy()
    frameHeight = frameOpenCvDnn.shape[0]
    frameWidth = frameOpenCvDnn[1]
    blob = cv2.dnn.blobFromImage(frameOpenCvDnn,1.0, (300,300), [102,117,123], True, False)
    net.setInput(blob)
    detection = net.forward()
    bboxes = []
    for i in range(detection.shapes[2]):
        confidence = detection[0,0,1,2]
        if confidence > conf_treshold:
            x1 = int(detection[0,0,i,3] * frameWidth)
            y1 = int(detection[0,0,i,4] * frameHeight)
            x2 = int(detection[0,0,i,5] * frameWidth)
            y2 = int(detection[0,0,i,6] * frameHeight)
            bboxes.append([x1,y1,x2,y2])
    return frameOpenCvDnn, bboxes


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