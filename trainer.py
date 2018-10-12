import cv2,os
import numpy as np
from PIL import Image
path = 'facedata'
recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
 
    imagePaths=[os.path.join(path,dataset) for dataset in os.listdir(path)] 
   
    faceSamples=[]

    Ids=[]
   
    for imagePath in imagePaths:

        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids
faces,Ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.write('/home/pi/facedata/trainner/trainner.yml')
