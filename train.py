import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.createLBPHFaceRecognizer();
path='d/dataSet'

def getImagesWithID(path):
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	IDs=[]
	for imagePath in imagePaths:
		faceImg=cv2.imread(imagePath);
		faceNp=cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		print ID
		IDs.append(ID)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
	return IDs,faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()
