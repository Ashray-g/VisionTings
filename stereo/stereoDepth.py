import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("leftPath")
parser.add_argument("rightPath")
parser.add_argument("stereoCalibFilePath")
args = parser.parse_args()

imgL = cv2.imread(args.leftPath,0)
imgR = cv2.imread(args.rightPath,0)

size = (np.shape(imgL)[1], np.shape(imgL)[0])

stereo_file = cv2.FileStorage(args.stereoCalibFilePath, cv2.FILE_STORAGE_READ)


K1 = stereo_file.getNode("K1").mat()
D1 = stereo_file.getNode("D1").mat()

K2 = stereo_file.getNode("K2").mat()
D2 = stereo_file.getNode("D2").mat()

R = stereo_file.getNode("R").mat()
T = stereo_file.getNode("T").mat()
E = stereo_file.getNode("E").mat()
F = stereo_file.getNode("F").mat()
R1 = stereo_file.getNode("R1").mat()
R2 = stereo_file.getNode("R2").mat()
P1 = stereo_file.getNode("P1").mat()
P2 = stereo_file.getNode("P2").mat()
Q = stereo_file.getNode("Q").mat()

stereo_file.release()

leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
left_rectified = cv2.remap(imgL, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
right_rectified = cv2.remap(imgR, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

imgRight = right_rectified
imgLeft = left_rectified

# Aditya tuned these
minDisparity = -1
numDisparities = 5 * 16
blockSize = 3
disp12MaxDiff = 12
uniquenessRatio = 10
speckleWindowSize = 50
speckleRange = 32
preFilterCap = 63
window_size = 3


stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               P1=8 * 3 * window_size,
                               P2=32 * 3 * window_size,
                               disp12MaxDiff = disp12MaxDiff,
                               uniquenessRatio = uniquenessRatio,
                               speckleWindowSize = speckleWindowSize,
                               speckleRange = speckleRange,
                               preFilterCap = preFilterCap,
                               mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
                               )


right_matcher = cv2.ximgproc.createRightMatcher(stereo)
lmbda = 8000
sigma = 1.5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
displ = stereo.compute(imgLeft, imgRight)
dispr = right_matcher.compute(imgRight, imgLeft)
displ = np.int16(displ)
dispr = np.int16(dispr)
disparity = wls_filter.filter(displ, imgLeft, None, dispr)

disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

cv2.imshow("disparity", disparity)
cv2.waitKey()
