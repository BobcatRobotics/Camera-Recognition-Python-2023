import cv2
from matplotlib import pyplot as plt
import numpy as np

path = "/content/c2.jpg"
img = cv2.imread(path)
     

# Меняем цветовое пространство BGR на RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_HSV = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
plt.imshow(img_HSV)
plt.show()

# Define lower and upper threshold values for yellow color in HSV color space
lower_yellow = np.array([22, 93, 0])
upper_yellow = np.array([45, 255, 255])

lower_purple = np.array([130, 50, 50])
upper_purple = np.array([160, 255, 255])
# Create a binary image with only yellow regions

img_thresh_p = cv2.inRange(img_HSV, lower_purple, upper_purple)
img_thresh_y = cv2.inRange(img_HSV, lower_yellow, upper_yellow)

plt.imshow(img_thresh_y)
plt.show()
kernel = np.ones((5, 5))
img_thresh_opened = cv2.morphologyEx(img_thresh_p, cv2.MORPH_OPEN, kernel)
cv2.imwrite("/content/thresh.jpg", img_thresh_y)
    
   
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply threshold to obtain a binary image
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Extract the largest 4 contours (excluding the background contour)
active_regions = []
for cnt in contours:
    # if len(active_regions) == 4:
    #     break
    if cv2.contourArea(cnt) > 1000: # exclude small contours
        x,y,w,h = cv2.boundingRect(cnt)
        active_regions.append(img[y:y+h,x:x+w])

# Draw rectangles around the extracted contours
for cnt in contours[:6]:
    if cv2.contourArea(cnt) > 1000: # exclude small contours
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow(img)
