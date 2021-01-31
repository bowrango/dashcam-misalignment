
import numpy as np
import cv2 as cv

from sklearn.neural_network import MLPRegressor

# ~~~SUPERVISED LEARNING~~~

# put labeled angle data into a Fx2 array, where F is the number of frames
y = np.loadtxt(f"labeled/{0}.txt")

# put cropped grey-scale images into a 2D array
# each row represents the features for the image
cap = cv.VideoCapture('tests/0.mp4')
X = []

while cap.isOpened():
    # full color image
    success, frame = cap.read()
    if frame is None:
        break
    cropped_frame = frame[1:100, 1:100]
    # cropped grey image
    cropped_grey = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
    cv.imshow('grey', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # flatten cropped image because MLPClassifier only works with 2D data
    flat_grey = cropped_grey.flatten()
    X.append(flat_grey)

# Shape X: (1200, 9801)
# Shape y: (1200, 2)

clf = MLPRegressor()
clf.fit(X, y)
