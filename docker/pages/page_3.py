import streamlit as st
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import speech_recognition as sr
import mediapipe as mp
import tensorflow as tf
import os.path
from concurrent.futures import ThreadPoolExecutor
import threading

st.markdown("# Voice 🎉")
st.sidebar.markdown("# Your voice as a remote 🎉")
st.markdown('Please take your time.  Say `webcam` to activate the webcam. The video should appear.  Say `photo` to take a picture.  Say `stop` to stop the webcam')

if os.path.isfile('photo.jpg'):
    with open('photo.jpg', "rb") as file:
        btn = st.download_button('Download your jpg',
                    data=file.read(),
                    file_name='photo.jpg',
                    mime="image/png")
else:
    st.write('No picture yet...')








# Créer une classe pour la webcam
class Webcam():
    def __init__(self):
        self._cap = cv2.VideoCapture(0)
        self._stop_flag = False
        self.frame = None

        

    def start(self):
        while not self._stop_flag:

            ret, self.frame = self._cap.read()
            if not ret:
                break
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            try:
                cv2.imshow('Webcam', self.frame)
            except:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._cap.release()
        cv2.destroyAllWindows()


    def stop(self):
        self._stop_flag = True


    def take_picture(self):
        file_name = 'photo.jpg'
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

