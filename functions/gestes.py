import cv2
import mediapipe as mp




mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def finger(frame):        
  

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:


            frame.flags.writeable = False
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(frame)


            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fingerCount = 0
            if results.multi_hand_landmarks:

              for hand_landmarks in results.multi_hand_landmarks:

                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label


                handLandmarks = []


                for landmarks in hand_landmarks.landmark:
                  handLandmarks.append([landmarks.x, landmarks.y])


                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                  fingerCount = fingerCount+1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                  fingerCount = fingerCount+1

                if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
                  fingerCount = fingerCount+1
                if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
                  fingerCount = fingerCount+1
                if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
                  fingerCount = fingerCount+1
                if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
                  fingerCount = fingerCount+1
                    
              return fingerCount


if __name__=='__main__':

    cap = cv2.VideoCapture(0)
    recording = False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('gestes.mp4', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        
    
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
            continue

        fingerCount = finger(frame)

        print(fingerCount)
        if fingerCount == 5 and not recording:
            recording = True


        elif fingerCount == 2 and recording:
            recording = False
            out.release()

        elif fingerCount == 10:
          cv2.imwrite('gestes.jpg', frame)


        if recording:
            out.write(frame)

        cv2.putText(frame, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)


        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(5) & 0xFF == 27: #esc 
            break
    cap.release()
    cv2.destroyAllWindows()


