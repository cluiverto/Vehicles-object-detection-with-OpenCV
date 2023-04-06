import cv2
import numpy as np 

cap = cv2.VideoCapture("Brasil6.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=3000, varThreshold=500, detectShadows=False, )

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape


    mask = object_detector.apply(frame)
    kernel = np.ones((10,10), np.uint8)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 2500:
            x, y, w,  h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (250, 0, 150), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)


    key = cv2.waitKey(20)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()