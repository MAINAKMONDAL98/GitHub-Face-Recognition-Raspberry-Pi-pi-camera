from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2
import time
import os
import sys
import RPi.GPIO as GPIO

#Setup Pins
pin = 14
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT, initial=0)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

display_window = cv2.namedWindow("Faces")

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('/home/pi/facedata/trainer/trainer.yml')



#face classifier
pathtoface =  "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(pathtoface)


font = cv2.FONT_HERSHEY_SIMPLEX




time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #FACE DETECTION STUFF
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        Id = recognizer.predict(gray[y:y+h,x:x+w])
        print Id
    
        if(Id[0] == 1):
          Id = "cr7"
        elif(Id[0] == 2):
            Id = "Samantha"
 
            
        cv2.rectangle(image,(x-22,y-90),(x+w+22,y-22),(0,255,0),-1)
        cv2.putText(image,str(Id),(x,y-40),font,2,(255,255,255),3)

  
    

    #DISPLAY TO WINDOW
    cv2.imshow("Faces", image)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if  key == ord("q"):
        camera.close()
        cv2.destroyAllWindows()
        break
