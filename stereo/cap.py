import cv2

cap = cv2.VideoCapture(1)
# right.open()

while(True):
    ret, frame1 = cap.read()
    print(ret)
    cv2.imshow("right", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()