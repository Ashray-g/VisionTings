import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    k = np.load("calibration_matrix.npy")
    d = np.load("distortion_coefficients.npy")

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

    if ids is not None and len(ids) > 0:
        # fig, ax = plt.subplots()
        x = np.array([])
        y = np.array([])
        for i in range(0, len(ids)):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, k, d)
            (rvecs - tvecs).any()
            cv2.aruco.drawAxis(frame,
                               k,
                               d,
                               rvecs,
                               tvecs,
                               0.06)
            # print("ves", tvecs)

            coords = (int(tvecs[0][0][0] * 1000 + 1), int(tvecs[0][0][1] * 1000 + 1))
            cv2.circle(frame, coords, 2, color=(0, 0, 255), thickness=5)


        # ax.scatter(x, y, s=len(x), vmin=0, vmax=100)

        # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #        ylim=(0, 8), yticks=np.arange(1, 8))

        # plt.show()


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
