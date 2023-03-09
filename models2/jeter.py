import speech_recognition as sr
import cv2

def listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen()

    try:
        text = r.recognize_google(audio, language='fr-FR')
        print(text)
    except:
        pass


while True:
    listen()