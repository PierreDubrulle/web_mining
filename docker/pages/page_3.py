import streamlit as st
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import mediapipe as mp
import tensorflow as tf

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")

import cv2
import speech_recognition as sr
import threading
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from concurrent.futures import ThreadPoolExecutor
import time







def age_genre(frame):

    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_net = cv2.dnn.readNetFromCaffe("models2/gender_deploy.prototxt", "models2/gender_net.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe("models2/age_deploy.prototxt", "models2/age_net.caffemodel")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Boucle sur chaque visage détecté
    for (x, y, w, h) in faces:
        # Extraction du visage
        face = frame[y:y+h, x:x+w]
        roi = cv2.resize(face, (48, 48))

        # Prétraitement du visage pour l'alimentation du modèle de reconnaissance de genre et d'âge
        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

        # Estimation du genre
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = "Homme" if gender_preds[0][0] > gender_preds[0][1] else "Femme"

        # Estimation de l'âge
        age_net.setInput(blob)
        age_preds = age_net.forward()
        max_index = np.argmax(age_preds[0])
        age = ageList[max_index]

        # Affichage des résultats sur l'image
        label = f"{gender}, {age} ans"
        return label
    return 'Non détecté'


names = ['antoine', 'axel','christelle_cor', 'christelle_kie', 'chrystelle','fatimetou',
            'florian','hugo','ibtissam','kenan','laura','leo','loic','louison','martin','matthieu',
            'nawres','pauline','pierre','robin','samuel','tho','titouan']

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
# Chargement du modèle de détection de visages de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Chargement du modèle de reconnaissance de genre et d'âge de OpenCV



class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
    
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]



def sentiment(frame):

    model = FacialExpressionModel("model.json", "model_weights.h5")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Extraction du visage
        face = gray[y:y+h, x:x+w]
        roi = cv2.resize(face, (48, 48))
        try:
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            return pred
        except:
            pred = 'Emotion non reconnue'
            return pred
    return 'Visage non détecté'
  

def recognition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Boucle sur chaque visage détecté
    for (x, y, w, h) in faces:
        # Extraction du visage

        # Prétraitement du visage pour l'alimentation du modèle de reconnaissance de genre et d'âge
        #blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100 and confidence >= 70):
                nom = names[id-1]
                confidence = "  {0}%".format(round(100 - confidence))
                return [nom, confidence, x, y, w, h]
        else:
                nom = 'Inconnu'
                confidence = "  {0}%".format(round(100 - confidence))
                return [nom, confidence, x, y, w, h]
    return ['Visage non détecté', '0%', 0, 0, 0, 0] 
    


def eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
    
    for (ex, ey, ew, eh) in eyes:
            coor_eyes =  [(ex, ey), (ex + ew, ey + eh)]
            return coor_eyes
    else:
        coor_eyes = [(0,0),(0,0)]
        return coor_eyes


def smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smile = smileCascade.detectMultiScale(
            gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(25, 25),
            )
    
    for (xx, yy, ww, hh) in smile:
            coor_smile = [(xx, yy), (xx + ww, yy + hh)]
            return coor_smile  
    else:
        coor_smile = [(0,0),(0,0)]
        return coor_smile  

  
    

def main(frame):
    executor = ThreadPoolExecutor(max_workers=2)

    # image = cv2.imread('Photo le 09-03-2023 à 05.53.jpg')
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    emotion_future = executor.submit(sentiment, frame)
    face_future = executor.submit(recognition, frame)
    # eye_future = executor.submit(eyes, frame)
    # smile_future = executor.submit(smile, frame)
    gender_age_future = executor.submit(age_genre, frame)

    emotion = emotion_future.result()
    nom, confidence, x, y, w, h = face_future.result()
    #cooreyes = eye_future.result()
    #coorsmile = smile_future.result()
    label = gender_age_future.result()


    return emotion, nom, confidence, x, y, w, h, label        


    # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.putText(frame, nom, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.putText(frame, confidence, (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.putText(frame, emotion, (x, y - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)




# Créer une classe pour la webcam
class Webcam():
    def __init__(self):
        self._cap = cv2.VideoCapture(0)
        self._stop_flag = False
        self.FRAME_WINDOW = st.image([])

    def start(self):
        while not self._stop_flag:

            ret, self.frame = self._cap.read()
            
            emotion, nom, confidence, x, y, w, h, label = main(self.frame)
            cv2.putText(self.frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(self.frame, nom, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(self.frame, confidence, (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(self.frame, emotion, (x, y - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.imshow('Webcam', self.frame)
            self.FRAME_WINDOW.image(self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._stop_flag = True

    def take_picture(self):
        file_name = './photo.jpg'
        cv2.imwrite(file_name, self.frame)


# Fonction pour traiter les commandes vocales
def process_commands(webcam):
    # Initialisation du Recognizer
    r = sr.Recognizer()

    # Boucle infinie pour écouter en continu
    while True:
        with sr.Microphone() as source:
            print("En attente d'une commande...")
            audio = r.listen(source)

            try:
                # Reconnaissance vocale
                command = r.recognize_google(audio, language='fr-FR')
                print("Commande détectée: " + command)

                # Traitement des commandes
                if 'webcam' in command:
                    # Démarrer la webcam dans un nouveau thread
                    webcam_thread = threading.Thread(target=webcam.start)
                    webcam_thread.start()
                elif 'stop' in command:
                    # Envoyer un signal à la webcam pour l'arrêter
                    webcam.stop()
                    break
                elif 'photo' in command:
                    photo_thread = threading.Thread(target=webcam.take_picture)
                    photo_thread.start()


            except sr.UnknownValueError:
                print("Impossible de comprendre la commande")
            except sr.RequestError as e:
                print("Erreur lors de la reconnaissance vocale; {0}".format(e))
        time.sleep(1)
if __name__ == '__main__':


    # Créer une instance de la webcam
    webcam = Webcam()

    # Lancer le processus pour traiter les commandes vocales
    process_thread = threading.Thread(target=process_commands, args=(webcam,))
    process_thread.start()

    # Attendre que le processus se termine
    process_thread.join()

    # Fermer la fenêtre de la webcam
    cv2.destroyAllWindows()
