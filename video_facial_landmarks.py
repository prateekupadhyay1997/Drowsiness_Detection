
import datetime
import time
import cv2
import dlib
import argparse
import imutils
import numpy as np
ap=argparse.ArgumentParser()
#ap.add_argument("-i","--webcam",type=int,default=-1,help="webcam input")
ap.add_argument("-p","--shape-predictor",required=True,help="facial landmark predictor")
args=vars(ap.parse_args())

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shape_predictor"])

cap=cv2.VideoCapture(0)

while(True):
    ret,frame=cap.read()
    frame=imutils.resize(frame,width=500)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    for rect in rects:
        shape=predictor(gray,rect)
        shape1=np.zeros((68,2),dtype="int64")
        #converting the shape list of co-ordinates into
        #numpy array of tuples containing co-ordinates

        for mj in range(0,68):
            shape1[mj]=(shape.part(mj).x,shape.part(mj).y)
        shape=shape1

        for (x,y) in shape:
            cv2.circle(frame,(x,y),1,(0,0,255),-1)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF

    if(key==ord('q')):
        break
