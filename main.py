# === OBJECTIVE ===

# Deliver 5 labels called 5.txt to 9.txt.
# These labels should be the 2D array of pitch and yaw angles of direction of the travel in camera frame
# We need to predict the pitch and yaw angles by which the openpilot camera is mis-aligned from the car frame.

# === APPROACH ===

# Let's try and implement optical flow to get a vectorized idea of how the scenery is changing as the car moves
# Perhaps we can then convert these to radians using something hacky :)

# ===  REFERENCES ===

# We are in the camera frame, what is the mis-alignment of the car frame, which is the direction of travel.

# Device Frame (Camera Frame): aligned with the road-facing camera used by openpilot.
# Car Frame: aligned with the car's direction of travel and road plane when going straight on a flat road

# The origin of the car frame is defined to be directly below the device frame, such that it is on the road plane.
# The orientation of this frame is not always aligned with the direction of travel or the road plane
# Suspension and other effects can mis-align the two frames while driving

# Images need to be in calibrated frame; defined to be aligned with car frame in pitch and yaw, and aligned with device frame in roll.
# The origin is the same as the device frame

# Tests .txt files will be run from a set directory:
# https://www.jetbrains.com/help/pycharm/performing-tests.html#Performing_Tests-11-procedure


import numpy as np
import cv2 as cv

# See if there are better video formats, this is super slow to convert online
cap = cv.VideoCapture('tests/0.mp4')

# ~~~Dense Optical Flow~~~

# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255
#
# while(1):
#     ret, frame2 = cap.read()
#     next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
#
#     flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#     mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
#     rgb = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
#
#     cv.imshow('frame2',rgb)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png',frame2)
#         cv.imwrite('opticalhsv.png',rgb)
#     prvs = next
#
# cap.release()
# cv.destroyAllWindows()

# ~~~Lucas-Kanade Optical Flow~~~
#   params for ShiTomasi corner detection
feature_params = dict(maxCorners=1,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for tracking
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while 1:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
