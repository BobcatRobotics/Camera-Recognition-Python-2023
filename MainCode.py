import cv2
import sys
from cscore import CameraServer
import numpy as np
print('Successful')
cs = CameraServer.getServer()
# Get user supplied values
# imagePath = sys.argv[1]
#imagePath = r"C:\Users\BobcatUser\Downloads\StreetImageTest.jpg"
camera = CameraServer.startAutomaticCapture(dev=1)

cascPath = "haarcascade_frontalface_default.xml"
cvSink = CameraServer.getVideo()
img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)
print('Camera Found?')
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
while True:
    # Read the image
    #image = cv2.imread(camera)
    time, img = cvSink.grabFrame(img)
    print('While Loop Successful')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Baby Monitor 2: Electric Boogaloo", img)
    cv2.waitKey(0)
    # Detect faces in the image 
    faces = faceCascade.detectMultiScale(
        #gray,
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print('Stuff Happened')
    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Are There Thinsg", img)
    cv2.waitKey(0)
    print('Successful')