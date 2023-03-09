import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import mediapipe as mp
import tensorflow as tf



names = ['antoine', 'axel','christelle_cor', 'christelle_kie', 'chrystelle','fatimetou',
            'florian','hugo','ibtissam','laura','leo','loic','louison','martin','matthieu',
            'pauline','pierre','robin','samuel','tho','titouan']
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



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




class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer.yml')
        self.cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.model = FacialExpressionModel("model.json", "model_weights.h5")
        self.nom = 'Inconnu'
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.padding=20
        self.font = cv2.FONT_HERSHEY_DUPLEX
        faceProto="./models2/opencv_face_detector.pbtxt"
        faceModel="./models2/opencv_face_detector_uint8.pb"
        ageProto="./models2/age_deploy.prototxt"
        ageModel="./models2/age_net.caffemodel"
        genderProto="./models2/gender_deploy.prototxt"
        genderModel="./models2/gender_net.caffemodel"

        self.MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
        self.ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList=['Male','Female']

        self.faceNet=cv2.dnn.readNet(faceModel,faceProto)
        self.ageNet=cv2.dnn.readNet(ageModel,ageProto)
        self.genderNet=cv2.dnn.readNet(genderModel,genderProto)

    def __del__(self):
        self.video.release()

    def highlightFace(self, net, frame, conf_threshold=0.7):
        frameOpencvDnn=frame.copy()
        frameHeight=frameOpencvDnn.shape[0]
        frameWidth=frameOpencvDnn.shape[1]
        blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*frameWidth)
                y1=int(detections[0,0,i,4]*frameHeight)
                x2=int(detections[0,0,i,5]*frameWidth)
                y2=int(detections[0,0,i,6]*frameHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn,faceBoxes

    def gender_age(self,frame):
    
        self.resultImg, self.faceBoxes=self.highlightFace(self.faceNet,frame)

        if not self.faceBoxes:
            print("No face detected")

        for faceBox in self.faceBoxes:
            self.face=frame[max(0,faceBox[1]-self.padding):
                    min(faceBox[3]+self.padding,frame.shape[0]-1),max(0,faceBox[0]-self.padding)
                    :min(faceBox[2]+self.padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(self.face, 1.0, (227,227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            self.genderPreds=self.genderNet.forward()
            self.gender=self.genderList[self.genderPreds[0].argmax()]

            self.ageNet.setInput(blob)
            self.agePreds=self.ageNet.forward()
            self.age=self.ageList[self.agePreds[0].argmax()]
        return self.age, self.gender


    # def finger(self,frame):

    #     with self.mp_hands.Hands(
    #         model_complexity=0,
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5) as hands:


    #             frame.flags.writeable = False
    #             frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    #             results = hands.process(frame)

    #             # Draw the hand annotations on the frame.
    #             frame.flags.writeable = True
    #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #             # Initially set finger count to 0 for each cap
    #             fingerCount = 0

    #             if results.multi_hand_landmarks:
                

    #                 for hand_landmarks in results.multi_hand_landmarks:
    #                     # Get hand index to check label (left or right)
    #                     handIndex = results.multi_hand_landmarks.index(hand_landmarks)
    #                     handLabel = results.multi_handedness[handIndex].classification[0].label

    #                     # Set variable to keep landmarks positions (x and y)
    #                     handLandmarks = []
                        
    #                     # Fill list with x and y positions of each landmark
    #                     for landmarks in hand_landmarks.landmark:
                        
                        
                        
    #                         handLandmarks.append([landmarks.x, landmarks.y])

    #                     # Test conditions for each finger: Count is increased if finger is 
    #                     #   considered raised.
    #                     # Thumb: TIP x position must be greater or lower than IP x position, 
    #                     #   deppeding on hand label.
    #                     if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
    #                         fingerCount = fingerCount+1
    #                     elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
    #                         fingerCount = fingerCount+1

    #                     # Other fingers: TIP y position must be lower than PIP y position, 
    #                     #   as frame origin is in the upper left corner.

    #                     if handLandmarks[8][1] < handLandmarks[6][1]:      #Index finger
    #                         fingerCount = fingerCount+1
    #                     if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
    #                         fingerCount = fingerCount+1
    #                     if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
    #                         fingerCount = fingerCount+1
    #                     if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
    #                         fingerCount = fingerCount+1

                        
    #     return fingerCount



    def get_frame(self):
        
        while (self.video.isOpened()):

            success, fr = self.video.read()



            return fr


    def visage_recognition(self, frame):

        fr = frame

        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            id, self.confidence = self.recognizer.predict(gray_fr[y:y+h,x:x+w])

            try:
                self.pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            except:
                self.pred = 'Emotion non reconnue'
                
            if (self.confidence < 100):
                try:
                    self.nom = names[id-1]
                    self.confidence = "  {0}%".format(round(100 - self.confidence))
                except:
                    pass
            else:
                try:
                    self.confidence = "  {0}%".format(round(100 - self.confidence))
                except:
                    pass
            

        return self.nom, self.confidence,self.pred, x, y, h, w

def gen(camera):

    recording = False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test.mp4', fourcc, 20.0, (640, 480))
    while True:
        
        try:
            frame = camera.get_frame()
        except:
            print('Error')
        # try:
        #     fingerCount = camera.finger(frame)
        # except:
        #     yield 'Error'
        try:
            nom, confidence,prediciton, x, y, h, w= camera.visage_recognition(frame)
        except:
            yield 'Error'
        try:
            age, gender = camera.gender_age(frame)
        except:
            yield 'Error'

        

        # cv2.putText(frame, str(fingerCount), (50, 450), camera.font, 3, (255, 0, 0), 10)

        cv2.putText(
                frame, 
                str(nom), 
                (x+5,y-5), 
                camera.font, 
                1, 
                (255,255,255), 
                2
            )

        cv2.putText(
                frame, 
                str(confidence), 
                (x+5,y+h-5), 
                camera.font, 
                1, 
                (255,255,0), 
                1
                )

            
        cv2.putText(frame, prediciton, (x+100, y+200), camera.font, 1, (255, 255, 0), 2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(frame,
                str(age),
                (x-100,y-100),
                camera.font,
                1,
                (255,255,255),
                2)
        
        cv2.putText(frame,
                str(gender),
                (x-100,y-200),
                camera.font,
                1,
                (255,255,255),
                2)


        print(recording)
        cv2.imshow('Facial Expression Recognization',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



if __name__=='__main__':
    # import os 
    # os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    # import tensorflow as tf

    gen(VideoCamera())





