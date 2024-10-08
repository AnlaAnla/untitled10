import cv2

backSub_mog = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgMask = backSub_mog.apply(frame)

    cv2.imshow("Input Frame", frame)
    cv2.imshow("Foreground Mask", fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == "q" or keyboard == 27:
        break
cv2.destroyAllWindows()