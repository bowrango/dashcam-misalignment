
Recent Work
------

- A nice article on sensor calibration, a few cool points:
- https://medium.com/lyftself-driving/high-fidelity-sensor-calibration-for-autonomous-vehicles-6af06eba4c26
  
- A paper covering the impact of camera misalignment on lane keeping assist systems:
  https://www.tandfonline.com/doi/full/10.1080/15472450.2020.1822174

- A paper on misalignment correction for depth estimation:   
  https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/4/562/files/2017/01/santoro-02.pdf

- A vehicle detection system based on Haar and Triangle features used for grouping taillights:
  https://ieeexplore.ieee.org/abstract/document/5164288/figures#figures
  
- This paper provides a method to calculate the misalignment of the VCS w.r.t the CCS. See Algorithm 3.
  http://web.cse.ohio-state.edu/~sinha.43/publications/conf/ipsn19-smartdashcam.pdf
 
  * estimates use forwards and lateral vanishing point (FVP) (LVP) 
  * manual calibration is the ground truth
  * angle predictions have errors less than 7 degrees. Mean is ~2 deg.

- Blog post from comma.ai about lateral planning:
  https://blog.comma.ai/end-to-end-lateral-planning/
 
 Methodology 
------

From the research thus far it seems the simplist approach is to track vanishing points in the image and relate those to the relative rotation of the camera.

- Paper about localizing position from a single image using pre-labeled data. Some nice graphics. The most relevent bit to us is Equation 3. 
  https://arxiv.org/pdf/2003.10664.pdf

- Slideshow on camera geometry. Descibes method to get rotation from vanishing point:
  https://www.cs.princeton.edu/courses/archive/fall13/cos429/lectures/11-epipolar

- Calculation of rotation matrix from vanishing points and focal length; see (6):
  https://annals-csis.org/proceedings/2012/pliks/110.pdf

The intrinsic matrix K projects a point in camera coordinates onto the image plane. It requires three parameters:

  * focal length of camera 
  * pixel width and height, the physical dimension of each pixel 
  * the image center 

  1. Solve for K using three orthogonal vanishing points
  2. Get rotation directly from vanishing points once K is known

Consider Convolutional Neural Networks (CNN)
------

A perspective transformation can be used to model yaw and pitch misalignments as the camera moves. The 3D perspective mapping can be modelled as a modified rotation matrix as seen in [this paper](https://ghassanalregibdotcom.files.wordpress.com/2016/10/santoro2012_mmsp1.pdf) as Equation (5). We can assume there is a perspective change across each consecutive frame, and that the roll is always 0 deg. because the camera is fixed to the vehicle. The proposed architecture would utilize a CNN to estimate the rotation transformation parameters: 

 * This problem could be formulated as a classification task (0 - 360 deg.)
 * Separate layers for pitch and yaw angles?
 * Transfer learning with ResNet50 as shown in [this article](https://d4nst.github.io/2017/01/12/image-orientation/)
 * Fine tune on labeled training data

[This paper](https://arxiv.org/pdf/1611.04298.pdf) might be a helpful aid. 


