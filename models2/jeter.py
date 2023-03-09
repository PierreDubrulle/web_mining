import cv2
import speech_recognition as sr
import multiprocessing as mp

# Créer une classe pour la webcam
class Webcam():
    def __init__(self, stop_flag):
        self._cap = cv2.VideoCapture(0)
        self._stop_flag = stop_flag

    def start(self):
        while not self._stop_flag.value:
            ret, frame = self._cap.read()
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._cap.release()
        cv2.destroyAllWindows()

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
                if 'oui' in command:
                    # Démarrer la webcam dans un nouveau processus
                    stop_flag = mp.Value('i', False)
                    webcam = Webcam(stop_flag) # création d'une instance de la webcam avec le stop_flag
                    p = mp.Process(target=webcam.start)
                    p.start()
                elif 'stop' in command:
                    # Envoyer un signal à la webcam pour l'arrêter
                    stop_flag.value = True
                    p.join() # Attendre que le processus se termine
                    break

            except sr.UnknownValueError:
                print("Impossible de comprendre la commande")
            except sr.RequestError as e:
                print("Erreur lors de la reconnaissance vocale; {0}".format(e))

if __name__ == '__main__':
    # Créer une instance de la webcam
    webcam = Webcam(None)

    # Créer un nouveau processus pour traiter les commandes vocales
    p = mp.Process(target=process_commands, args=(webcam,))
    p.start()

    # Attendre que le processus se termine
    p.join()

    # Fermer la fenêtre de la webcam
    cv2.destroyAllWindows()
