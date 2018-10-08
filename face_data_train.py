from picamera.array import PiRGBArray
from picamera import PiCamera
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

#face classifier
pathtoface = os.path.join(sys.path[0], 'face_classifier.xml')
face_cascade = cv2.CascadeClassifier(pathtoface)
face_id = 2
count = 0
#full body classifier
pathtobody = os.path.join(sys.path[0], 'full_body.xml')
body_cascade = cv2.CascadeClassifier(pathtobody)

time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #FACE DETECTION STUFF
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        GPIO.output(pin, GPIO.HIGH)
                count += 1


        # Save the captured image into the datasets folder
        cv2.imwrite("/home/pi/facedata/dataset/s." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])


    #BODY DETECTION STUFF
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in bodies:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        GPIO.output(pin, GPIO.HIGH)
    #DISPLAY TO WINDOW
    cv2.imshow("Faces", image)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if count>26 or key == ord("q"):
        camera.close()
        cv2.destroyAllWindows()
        break
