import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load the video file
cap = cv2.VideoCapture(0)
#  Define lower and upper threshold values for yellow and purple colors in HSV color space
lower_yellow = np.array([22, 93, 0])
upper_yellow = np.array([45, 255, 255])

lower_purple =  np.array([110,50,50])
upper_purple =  np.array([130,255,255])

# Create a kernel for morphological opening
kernel = np.ones((5, 5))

while True:
    # Read a frame from the video file
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame from RGB to HSV color space
    frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Threshold the frame to extract yellow and purple regions
    frame_thresh_y = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    frame_thresh_p = cv2.inRange(frame_hsv, lower_purple, upper_purple)

    # Perform morphological opening on the thresholded images to remove noise
    frame_thresh_opened_y = cv2.morphologyEx(frame_thresh_y, cv2.MORPH_OPEN, kernel)
    frame_thresh_opened_p = cv2.morphologyEx(frame_thresh_p, cv2.MORPH_OPEN, kernel)

    # Find contours in the thresholded images
    contours_y, _ = cv2.findContours(frame_thresh_opened_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_p, _ = cv2.findContours(frame_thresh_opened_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order
    contours_y = sorted(contours_y, key=cv2.contourArea, reverse=True)
    contours_p = sorted(contours_p, key=cv2.contourArea, reverse=True)
    area = 1000

 
    # # Extract the largest contours (excluding the background contour)
    active_regions = []
    for cnt in contours_y + contours_p:
        if cv2.contourArea(cnt) > area: # exclude small contours
            x,y,w,h = cv2.boundingRect(cnt)
            active_regions.append(frame[y:y+h,x:x+w])

    # Draw rectangles around the extracted contours
    for cnt in contours_y + contours_p:
        if cv2.contourArea(cnt) > area: # exclude small contours
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
