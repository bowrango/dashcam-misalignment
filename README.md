
 
Motivation
------
The devices that run [openpilot](https://github.com/commaai/openpilot/) are not mounted perfectly. The camera
is not exactly aligned to the vehicle. There is some pitch and yaw angle between the camera of the device and
the vehicle, which can vary between installations. Estimating these angles is essential for accurate control
of the vehicle. More info  can be found in [this readme](https://github.com/commaai/openpilot/tree/master/common/transformations).

Methodology
------

1. Feature extraction from video using Canny edge detection:
    - cropped; grayscale; Gaussian (3,3) blur
    - Otsu's method for Canny thresholds
    - output is sparse array (per training video): shape (1200, 28618) 
    
2. Sequential training on RNN:
    - input tensor: shape (1, 1, 28618) (batch, time_step, input_size)
    - hidden state tensor: shape (1, 1, 28618) (n_layers, batch, hidden_size)
    - prediction tensor: shape(1, 1, 2) (batch, time_step, output_size)
    - Pytorch Adam optimizer; MSELoss objective (squared L2 norm)
    - 80/20 train test split


