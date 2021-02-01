import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor


# training data is focused on the boundary region of the car front and the road
# this should hopefully emphasis mis-alignments of the camera and car frames

# TODO: experiment with image processing
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

        # resize, greyscale, crop
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        shape = np.shape(frame_grey)
        cropped_frame = frame_grey[300:350, 200:400]

        # consider blurring?

        # roi
        rect = cv.rectangle(frame, (200, 300), (400, 350), (255, 0, 0), 2)
        if watch:
            cv.imshow('grey', cropped_frame)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        features.append(frame_grey.flatten())
    return np.asarray(features)


# all nan occurrences are replaced by the previous real angle value
# since angles frame by frame don't change much, the previous angle is a better approx than 0s
def read_angles(angles_f, ffill=0):
    # high precision isn't necessary
    arr = np.loadtxt(angles_f, dtype='float16')

    mask = np.isnan(arr)
    tmp = arr[0].copy()
    arr[0][mask[0]] = ffill
    mask[0] = False
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    out = np.take_along_axis(arr, np.maximum.accumulate(idx, axis=0), axis=0)
    arr[0] = tmp
    return out


def get_mse(gt, test):
    return np.mean(np.mean((gt - test) ** 2, axis=0))


# ~~~SUPERVISED LEARNING~~~

# train on videos 0-3, test on video 4
X = []
Y = []
for i in range(4):
    y = read_angles(f"labeled/{i}.txt")
    x = encode_features(f"tests/{i}.avi", watch=True)

    X.extend(x)
    Y.extend(y)

np.save('X', X)
np.save('Y', Y)

X = np.load('X.npy')
y = np.load('Y.npy')

# true_y = read_angles(f"labeled/{4}.txt")
# test_X = encode_features(f"tests/{4}.avi")

# TODO: Pre-cache video frames for debugging and consider resizing images
clf = MLPRegressor()
clf.fit(X, y)
plt.plot(clf.loss_curve_)
plt.show()

# pred_y = clf.predict(test_X)
# np.save('pred_y', pred_y)
#
# zero_mse = get_mse(true_y, np.zeros_like(true_y))
# mse = get_mse(true_y, pred_y)
#
# percent_err_vs_all_zeros = 100 * (mse / zero_mse)
# print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
