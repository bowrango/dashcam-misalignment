import cv2 as cv
import torch 
from torch import nn
import numpy as np

import pickle
import time
import matplotlib.pyplot as plt

from rnn import RNN


def auto_canny(image, sigma=0.33, watch=False):

    # Otsu's method for thresholding
    high_thresh, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    otsu_edge = cv.Canny(image, low_thresh, high_thresh)
    if watch:
        cv.imshow('otsu-canny', otsu_edge)

    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    auto_edge = cv.Canny(image, lower, upper)
    if watch:
        cv.imshow('auto-canny', auto_edge)
    return otsu_edge


def read_angles(angles_f, ffill=0):

    # returned angles are a (1200, 2) numpy array
    #   - values are NaN when vehicle speed is < ~4 m/s. 
    #   - replace these occurances with previous real angle value
    #   - assumption is the angle doesnt change much during low speeds 

    arr = np.loadtxt(angles_f, dtype=np.float32)
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
    i = [] 
    while cap.isOpened:
        success, frame = cap.read()
        if success:
            # === Canny Edge Detection for Tracking the Dash Horizon ==
            kernel = (3,3) # TODO: experiment with vert/horiz kernels
            canny_img = frame[tl_corner[0]:br_corner[0], tl_corner[1]:br_corner[1]]
            canny_gray = cv.cvtColor(canny_img, cv.COLOR_BGR2GRAY)
            canny_blur = cv.GaussianBlur(canny_gray, kernel, 0)

            edge = auto_canny(canny_blur, watch=watch)
            edge = np.divide(edge, 255)   # binary 0-1 sparse matrix
            intensity = np.sum(edge, axis=None)
            i.append(intensity)
            features.append(edge.flatten())

            # draw rect on video 
            if watch:
                #rect = cv.rectangle(frame, (tl_corner[1],tl_corner[0]), (br_corner[1],br_corner[0]), (255, 0, 0), 2)
                cv.imshow(f"{video_f}", frame)
            k = cv.waitKey(30) & 0xff

            if k == 27:
                break
        else:
            cap.release()
            break
    
    end = time.time()
    print(f"Processing Time: {round(end-start,2)} (s)")
    return np.asarray(features, dtype=np.float32), np.asarray(i, dtype=np.int16) # float32 for converting torch FloatTensor

# 1. Extract features from training videos using Canny edge detection
# 2. Read in labeled angles
# 3. Train RNN

if __name__ == "__main__":
    
    torch.manual_seed(1)    # reproducible
    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.02)   # optimize all cnn parameters
    loss_func = nn.MSELoss()
    h_state = None      # for initial hidden state

    # == training loop over 1 video(s) ==
    for i in range(4):
        X, intensity = extract_features(f'labeled/{i}.mp4', watch=True)
        Y = read_angles(f'labeled/{i}.txt')

        np.save(f"X{i}.npy", X)
        np.save(f"Y{i}.npy", Y)

        # X = np.load(f"X{i}.npy")
        # Y = np.load(f"Y{i}.npy")
        s = np.array(range(len(X)))
        
        train_mse = []

        n = X.shape[1]
        m = Y.shape[1]
        batch = 1
        time_step = 1

        theta = np.zeros(len(X))
        phi = np.zeros(len(X))
        #  one frame at a time
        for step in range(len(X)):
            
            x = X[step].reshape(batch, time_step, n) # shape (batch, time_step, input_size)
            x_t = torch.from_numpy(x)

            y = Y[step].reshape(batch, time_step, m)
            y_t = torch.from_numpy(y)

            prediction, h_state = rnn(x_t, h_state)  
            
            # !! next step is important !!
            h_state = h_state.data        # repack the hidden state, break the connection from last iteration

            loss = loss_func(prediction, y_t)
            optimizer.zero_grad()                   # clear gradients for this training step
            loss.backward()                         # backpropagation, compute gradients
            optimizer.step()                        # apply gradients

            train_mse.append(np.float32(loss.data))
            theta[step] = prediction.data.numpy().flatten()[0]
            phi[step] = prediction.data.numpy().flatten()[1]
        
    
        # === TRAINING RESULTS FOR EACH VIDEO ===
        plt.figure(1)
        plt.subplot(211)
        plt.plot(s, theta*(180/np.pi), 'r.') # in deg.
        plt.plot(s, Y[:,0]*(180/np.pi), 'b.') # groundtruth
        plt.ylabel('Pitch Angle (deg.)')
        # plt.ylim([1, 3])
        plt.legend(['predicted', 'groundtruth'])

        plt.subplot(212)
        plt.plot(s, phi*(180/np.pi), 'r.') # in deg.
        plt.plot(s, Y[:,1]*(180/np.pi), 'b.') # groundtruth
        plt.ylabel('Yaw Angle (deg.)')
        # plt.ylim([1, 3])
        plt.xlabel('Training Frame')    
        plt.show()

        plt.figure(2)
        plt.plot(s, intensity, 'k.')
        plt.ylabel('Edge Intensity')
        plt.xlabel('Frame')
        plt.show()

    # === testing  === 
    # X = np.load(f"X4.npy")
    # Y = np.load(f"Y4.npy")
    s = np.array(range(len(X)))
    
    test_mse = []
    n = X.shape[1]
    m = Y.shape[1]
    batch = 1
    time_step = 1

    theta = np.zeros(len(X))
    phi = np.zeros(len(X))
    #  one frame at a time
    for step in range(len(X)):
        
        x = X[step].reshape(batch, time_step, n) # shape (batch, time_step, input_size)
        x_t = torch.from_numpy(x)

        y = Y[step].reshape(batch, time_step, m)
        y_t = torch.from_numpy(y)

        prediction, h_state = rnn(x_t, h_state)  
        h_state = h_state.data
        test_mse.append(np.float32(loss.data))
        theta[step] = prediction.data.numpy().flatten()[0]
        phi[step] = prediction.data.numpy().flatten()[1]
        
    
    # === TESTING RESULTS ===
    plt.figure(1)
    plt.subplot(211)
    plt.plot(s, theta*(180/np.pi), 'r.') # in deg.
    plt.plot(s, Y[:,0]*(180/np.pi), 'b.') # groundtruth
    plt.ylabel('Pitch Angle (deg.)')
    plt.legend(['predicted', 'groundtruth'])

    plt.subplot(212)
    plt.plot(s, phi*(180/np.pi), 'r.') # in deg.
    plt.plot(s, Y[:,1]*(180/np.pi), 'b.') # groundtruth
    plt.ylabel('Yaw Angle (deg.)')
    plt.xlabel('Testing Frame')    
    plt.show()  


    
        