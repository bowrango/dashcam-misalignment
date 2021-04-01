
Frames of Reference
------

Video is from the perspective of the camera, what is the misalignment to the car frame? The camera frame and car frame have different directions of travel.

Device Frame (Camera Frame): aligned with the road-facing camera used by openpilot.
Car Frame: aligned with the car's direction of travel and road plane when going straight on a flat road

 - The origin of the car frame is defined to be directly below the device frame, such that it is on the road plane.
 - Camera frame and car frame will be misaligned due to installation and suspension effects

Openpilot needs images to be in a calibrated frame; defined to be aligned with car frame in pitch and yaw, and aligned with device frame in roll.

Welcome to the comma.ai Calibration Challenge!
======

Your goal is to predict the direction of travel (in camera frame) from provided dashcam video.

- This repo provides 10 videos. Every video is 1min long and 20 fps.
- 5 videos are labeled with a 2D array describing the direction of travel at every frame of the video
  with a pitch and yaw angle in radians.
- 5 videos are unlabeled. It is your task to generate the labels for them.
- The example labels are generated using a Neural Network, and the labels were confirmed with a SLAM algorithm.
- You can estimate the focal length to be 910 pixels.

Context
------
The devices that run [openpilot](https://github.com/commaai/openpilot/) are not mounted perfectly. The camera
is not exactly aligned to the vehicle. There is some pitch and yaw angle between the camera of the device and
the vehicle, which can vary between installations. Estimating these angles is essential for accurate control
of the vehicle. The best way to start estimating these values is to predict the direction of motion in camera
frame. More info  can be found in [this readme](https://github.com/commaai/openpilot/tree/master/common/transformations).

Deliverable
-----

Your deliverable is the 5 labels called 5.txt to 9.txt. These labels should be a 2D array that contains
the pitch and yaw angles of the direction of travel (in camera frame) of every frame of the respective videos.
Zip them up and e-mail it to givemeajob@comma.ai.


Evaluation
-----

We will evaluate your mean squared error against our ground truth labels. Errors for frames where the car
speed is less than 4m/s will be ignored. Those are also labeled as NaN in the example labels.

This repo includes an eval script that will give an error score (lower is better). You can use it to test your
solutions against the labeled examples. We will use this script to evaluate your solution. 


Hints
------
- Keep the goal and evaluation script in mind, creative solutions are allowed.
- Look at plots of your solutions before submitting.


$500 Prize
------
The first submission that scores an error under 25% on the unlabeled set, will receive a $500 prize.
