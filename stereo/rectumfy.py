# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# cv_file = cv2.FileStorage("left_cam.yml", cv2.FILE_STORAGE_READ)
# camera_matrix = cv_file.getNode("K").mat()
# dist_matrix = cv_file.getNode("D").mat()
# cv_file.release()
#
# imgLeft = cv2.imread('left0.png', 0)
#
# h1, w1 = imgLeft.shape
#
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
#
# _, H1, H2 = cv2.stereoRectifyUncalibrated(
#     np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
# )