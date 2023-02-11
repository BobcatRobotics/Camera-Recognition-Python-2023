# Import the camera server
from cscore import CameraServer

# Import OpenCV and NumPy
import cv2
import numpy as np
print('Startup')
def main():
    print('Main')
   
    print('camera')

    cs = CameraServer.getServer()

    print(cs.getConfigJson())

    # Capture from the first USB Camera on the system
    camera = CameraServer.startAutomaticCapture(dev=1)

    camera.setResolution(430, 350)


    # Capture from the first USB Camera on the system
    # camera = CameraServer.startAutomaticCapture()
    # camera.setResolution(320, 240)

    # Get a CvSink. This will capture images from the camera
    cvSink = CameraServer.getVideo()
    # print(cvSink)

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = CameraServer.putVideo("Name", 320, 240)

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    while True:
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        time, img = cvSink.grabFrame(img)
        cv2.imshow("Baby Monitor", img)
        cv2.waitKey(1)
        if time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError());
            # skip the rest of the current iteration
            continue

        #
        # Insert your image processing logic here!
        #

        # (optional) send some image back to the dashboard
        outputStream.putFrame(img)
main()