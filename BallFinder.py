import cv2
import numpy as np
cv2.imread ()
cv2.imshow('Original Image',rawImage) 
cv2.waitKey(0)

hsv = cv2.cvtColor(rawImage, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image',hsv)
cv2.waitKey(0)

hue ,saturation ,value = cv2.split(hsv)
cv2.imshow('Saturation Image',saturation)
cv2.waitKey(0)

retval, thresholded = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Thresholded Image',thresholded)
cv2.waitKey(0)

medianFiltered = cv2.medianBlur(thresholded,5)
cv2.imshow('Median Filtered Image',medianFiltered)
cv2.waitKey(0) 

cnts, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
# compute the center of the contour
    M = cv2.moments(c)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])


first = cv2.drawContours(rawImage, [c], -1, (0, 255, 0), 2)
second =cv2.circle(rawImage, (cX, cY),1 , (255, 255, 255), -1)


cv2.imshow('Objects Detected',rawImage)
cv2.waitKey(0)
