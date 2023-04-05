import cv2

cap = cv2.VideoCapture("Brasil6.mp4")

#Detector
object_detector = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=1500)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    #Extract ROI
    roi = frame[180:720, 500:900]

    #Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Setup area & remove small elements
        area = cv2.contourArea(c)
        if area > 100:
            #cv2.drawContours(roi, [c], -1, (250, 0, 150), 2)
            x, y, w,  h = cv2.boundingRect(c)
            cv2.rectangle(roi, (x,y), (x + w, y + h), (250, 0, 150), 3)



    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Roi", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()