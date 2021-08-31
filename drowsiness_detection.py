import cv2
import numpy as np
import dlib
import argparse
from collections import OrderedDict
import datetime
import time
from scipy.spatial import distance as dist
import playsound
from threading import Thread
def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear

ap=argparse.ArgumentParser()
ap.add_argument('-a',"--alarm",type=str,default="",help="path alarm .wav file")
ap.add_argument('-p',"--shape-predictor",required=True,help="dlib's facial landmark Predictor")

args=vars(ap.parse_args())

EYE_AR_THRESH=0.3
EYE_AR_CONSEC_FRAMES=80
COUNTER=0
ALARM_ON=False
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args['shape_predictor'])

FACIAL_LANDMARKS_IDXS=OrderedDict([("mouth",(48,68)),("right_eyebrow",(17,22)),("left_eyebrow",(22,27)),("right_eye",(36,42)),("left_eye",(42,48)),("nose",(27,35)),("jaw",(0,17))])

(lStart,lEnd)=FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart,rEnd)=FACIAL_LANDMARKS_IDXS["right_eye"]



cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    #height,width,layers=np.shape(frame)
    frame = cv2.resize(frame, (450,450))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    for rect in rects:
        shape=predictor(gray,rect)
        shape1=np.zeros((68,2),dtype="int64")

        for mj in range(0,68):
            shape1[mj]=(shape.part(mj).x,shape.part(mj).y)
        shape=shape1

        x=rect.left()
        y=rect.top()
        w=rect.right()-x
        h=rect.bottom()-y
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)
        ear=(leftEAR+rightEAR)/2.0
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)

        if ear < EYE_AR_THRESH:
            COUNTER=COUNTER+1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "COUNTER ={}".format(COUNTER), (260, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (260, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
