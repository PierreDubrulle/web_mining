import cv2
import speech_recognition as sr
import threading

# Créer une classe pour la webcam
class Webcam():
    def __init__(self):
        self._cap = cv2.VideoCapture(0)
        self._stop_flag = False

    def start(self):
        while not self._stop_flag:
            ret, frame = self._cap.read()
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._stop_flag = True

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
