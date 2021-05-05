import numpy as np
import cv2 as cv
import time

def auto_canny(image, sigma=0.33, watch=False):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    if watch:
        cv.imshow('autocanny', edged)
    return edged

def read_angles(angles_f, ffill=0):

    # returned angles are a (1200, 2) numpy array
    #   - values are NaN when vehicle speed is < ~4 m/s. 
    #   - replace these occurances with previous real angle value
    #   - assumption is the angle doesnt change much during low speeds 

    arr = np.loadtxt(angles_f, dtype='float16')
    mask = np.isnan(arr)
    tmp = arr[0].copy()
    arr[0][mask[0]] = ffill
    mask[0] = False
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    angles = np.take_along_axis(arr, np.maximum.accumulate(idx, axis=0), axis=0)
    arr[0] = tmp
    
    return angles

def extract_features(video_f, watch=False):
    start = time.time()

    # returned features are stored in a (1200, 28618) binary sparse array representing the edge of the dash across frames
    #   - region of interest is (41, 698). 41 * 698 = 28618 features per frame
    #   - training/test videos are 1 minute at 20 fps. 60 * 20 = 1200 observations per video
    features = []

    # take first frame and get dimensions 
    cap = cv.VideoCapture(video_f)
    ret, frame0 = cap.read()
    H, W, _ = frame0.shape
    cap.release()

    # rectangular region to track features
    # TODO: This hard-coded boundary is not good. Ideally this would be automaticaly identified. 
    tl_corner = (int(np.rint(5*H/6 - 20)), int(np.rint(W/5)))
    br_corner = (int(np.rint(H - 125)), int(np.rint(4*W/5)))

    cap = cv.VideoCapture(video_f)
    while cap.isOpened:
        success, frame = cap.read()
        if success:
            # === Canny Edge Detection for Tracking the Dash Horizon ==
            kernel = (1,5) # TODO: experiment with vert/horiz kernels
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            canny_img = frame_gray[tl_corner[0]:br_corner[0], tl_corner[1]:br_corner[1]]
            canny_img = cv.GaussianBlur(canny_img, kernel, 0)

            edges = auto_canny(canny_img, watch=watch)
            features.append(edges.flatten())

            # draw rect on video 
            if watch:
                rect = cv.rectangle(frame, (tl_corner[1],tl_corner[0]), (br_corner[1],br_corner[0]), (255, 0, 0), 2)
                cv.imshow(f"{video_f}", frame)
            k = cv.waitKey(30) & 0xff

            if k == 27:
                break
        else:
            cap.release()
            break
    
    end = time.time()
    print(f"Processing Time: {round(end-start,2)} (s)")
    return np.asarray(features)

# 1. Extract features from training videos using Canny edge detection
# 2. Read in labeled angles
# 3. Train RNN

if __name__ == "__main__":

    # TODO: should be able to handle .avi files, use .mp4 for now 
    X = extract_features("labeled/0.mp4", watch=True)
    y = read_angles("labeled/0.txt")
    
    print(X.shape)
    print(y.shape)