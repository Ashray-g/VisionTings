import cv2

captureLeft = cv2.VideoCapture(0)
captureRight = cv2.VideoCapture(1)

fourccRight = cv2.VideoWriter_fourcc(*'mp4v')
fourccLeft = cv2.VideoWriter_fourcc(*'mp4v')
videoWriterRight = cv2.VideoWriter('/Users/ashray/PycharmProjects/pythonProject4/right.mp4', fourccRight, 30.0, (640, 480))
videoWriterLeft = cv2.VideoWriter('/Users/ashray/PycharmProjects/pythonProject4/left.mp4', fourccLeft, 30.0, (640, 480))

for i in range(1, 300):

    ret1, left = captureLeft.read()
    ret2, right = captureRight.read()

    if ret1 and ret2:
        left = cv2.resize(left, (640, 480))
        right = cv2.resize(right, (640, 480))
        cv2.imshow('videoRight', right)
        cv2.imshow('videoLeft', left)
        videoWriterLeft.write(left)
        videoWriterRight.write(right)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
        break

captureLeft.release()
captureRight.release()
videoWriterLeft.release()
videoWriterRight.release()
cv2.destroyAllWindows()
