import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from concurrent.futures import ThreadPoolExecutor


st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")
st.title('Reconnaissance faciale')




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
            'florian','hugo','ibtissam','laura','leo','loic','louison','martin','matthieu',
            'pauline','pierre','robin','samuel','tho','titouan']

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
        face = frame[y:y+h, x:x+w]

        # Prétraitement du visage pour l'alimentation du modèle de reconnaissance de genre et d'âge
        blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
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

  
    

def main():
    # Initialisation de la capture vidéo de la webcam
    global cap
    cap = cv2.VideoCapture(0)
    executor = ThreadPoolExecutor(max_workers=2)

    # Boucle jusqu'à ce que l'utilisateur appuie sur la touche "q"
    while True:
    
        # Capture d'une image de la webcam
        ret, image = cap.read()
        if not ret:
            break
        # image = cv2.imread('Photo le 09-03-2023 à 05.53.jpg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        emotion_future = executor.submit(sentiment, image)
        face_future = executor.submit(recognition, image)
        eye_future = executor.submit(eyes, image)
        smile_future = executor.submit(smile, image)
        gender_age_future = executor.submit(age_genre, image)

        emotion = emotion_future.result()
        nom, confidence, x, y, w, h = face_future.result()
        cooreyes = eye_future.result()
        coorsmile = smile_future.result()
        label = gender_age_future.result()

    
            


        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, nom, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, confidence, (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        FRAME_WINDOW.image(image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Arrêt de la capture vidéo et fermeture de la fenêtre




st.title("Webcam Live Feed")
FRAME_WINDOW = st.image([])
run = st.checkbox('Run')
if run:
     main()


# camera = cv2.VideoCapture(0)
else:
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()