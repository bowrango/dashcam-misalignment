import numpy as np
import cv2 as cv

from sklearn.neural_network import MLPRegressor


def encode_features(video_f, watch=False):
    # put the cropped grey-scale images into a 2D array
    # each row represents the features for the image
    cap = cv.VideoCapture(video_f)
    features = []

    while cap.isOpened():
        # full color image
        success, frame = cap.read()
        if frame is None:
            break
        cropped_frame = frame[1:100, 1:100]
        # cropped grey image
        cropped_grey = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
        if watch:
            cv.imshow('grey', frame)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        features.append(cropped_grey.flatten())
    return features


# all nan occurrences are replaced by the previous real angle value
# since angles frame by frame don't change much, the previous angle is a better approx than 0s
def read_angles(angles_f, ffill=0):
    arr = np.loadtxt(angles_f)

    mask = np.isnan(arr)
    tmp = arr[0].copy()
    arr[0][mask[0]] = ffill
    mask[0] = False
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    out = np.take_along_axis(arr, np.maximum.accumulate(idx, axis=0), axis=0)
    arr[0] = tmp
    return out


# ~~~SUPERVISED LEARNING~~~

y = read_angles(f"labeled/{2}.txt")
X = encode_features(f"tests/{0}.avi")

true_y = read_angles(f"labeled/{1}.txt")
test_X = encode_features(f"tests/{1}.avi")

# handle for nan occurrences
# Shape X: (1200, 9801)
# Shape y: (1200, 2)

# TODO: Pre-cache video frame for debugging and consider resizing images
clf = MLPRegressor()
clf.fit(X, y)

print(clf.predict(test_X))
