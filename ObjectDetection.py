import cv2

imcap = cv2.VideoCapture(0)
imcap.set(3, 480)
imcap.set(4, 640)

while True:
    success, img = imcap.read()

    cv2.imshow('internal webcam', img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

imcap.release()
cv2.destroyWindow('internal webcam')

