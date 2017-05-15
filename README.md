## Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1.jpg "To be calibrated"
[image2]: ./output_images/undistorted.jpg "Undistorted Image"
[image3]: ./output_images/test4.jpg "Test Image"
[image4]: ./output_images/undist_test4.jpg "Undistorted Image"
[image5]: ./output_images/binary_image.jpg "Binary Image"
[image6]: ./output_images/src.JPG "Source Image"
[image7]: ./output_images/dst.JPG "Destination Image"
[image8]: ./output_images/warped_image.jpg "Warped Image"
[image9]: ./output_images/warped_bin_image.jpg "Warped Binary Image"
[image10]: ./output_images/hist.jpg "Histogram"
[image11]: ./output_images/sliding_window.jpg "Sliding Window"
[image12]: ./output_images/recast.png "Recast Image"


### Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

The code for this step is contained in the file called `camera_calibration.py`  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1] ![alt text][image2]
<p style="text-align: center;"> Figure 1: (left) Original Image (right) Undistorted Image </p>


### Pipeline (single images)

#### 1. Image correction for Camera distortion

After camera calibration matrix and distortion coefficients were obtained from previous step, all image frames in the video were corrected for distortion. To demonstrate this step, here's an example of a test image which was then undistorted using the pickle data containing the camera calibration matrix and the distortion coefficients.

![alt text][image3] ![alt text][image4]
<p style="text-align: center;"> Figure 2: (left) Test Image (right) Undistorted Test Image </p>

#### 2. Creating binary image using Color transforms and Sobel gradients

I used a combination of color and gradient thresholds to generate a binary image (gradient thresholding steps at lines 29 through 47 and color thresholding steps at lines 49 through 80 in `image_generator.py`).  Here's an example of my output for this step.  

![alt text][image5]
<p style="text-align: center;"> Figure 3: Binary Image </p>

#### 3. Perspective Transformation of Image

The code for my perspective transform appears in lines 47 through 65 in the file `main.py`. The `getPerspectiveTransform()` function takes as inputs as source (`src`) and destination (`dst`) points. Then I use `warpPerspective()` function to apply the warp transform on the filtered binary image from above step 2.

```python
	src = np.float32([[(0.5*img.shape[1] - 0.5*top_width*img.shape[1]),height*img.shape[0]], # Point 1
			[(0.5*img.shape[1] + 0.5*top_width*img.shape[1]),height*img.shape[0]], # Point 2
			[(0.5*img.shape[1] - 0.5*bottom_width*img.shape[1]),chop_bottom*img.shape[0]],# Point 3
			[(0.5*img.shape[1] + 0.5*bottom_width*img.shape[1]),chop_bottom*img.shape[0]],# Point 4
			])
	dst = np.float32([[offset,0],[img.shape[1]-offset,0],[offset,img.shape[0]],[img.shape[1]-offset,img.shape[0]]])
	
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 576, 504      | 128, 0        | 
| 704, 504      | 1152, 720     |
| 160, 676      | 128, 720      |
| 1120, 676     | 1152, 0       |

The following images show this step before applying the perspective transform.

![alt text][image6] ![alt text][image7]

<p style="text-align: center;"> Figure 4: (left) Source Points Image (right) Destination Points </p>

Then I applied the perspective transform on the same image as above to get a **Bird's-eye view** of the image an example of which is shown below.

![alt text][image8]![alt text][image9]!

<p style="text-align: center;"> Figure 5: (left) Source Points Image (right) Destination Points </p>

#### 4. Searching for Lane Lines

Once the transformed images were in the pipeline, the next step was to search for the lane lines in the images. This was accomplished using one of the two functions either `blind_search` or `not_blind_search` functions which are a part of the `image_generator.py` file starting on line 123 and 254 respectively. 

`blind_search` function starts to search for pixel values which might belong to the lane line from the bottom of the frames. This is done by taking histogram of the image and finding out the indices of the max values of histogram which would correspond to the lane lines as shown below.

![alt text][image10]
<p style="text-align: center;"> Figure 6: Histogram taken at the bottom of the image frame </p>

Then a technique called `the sliding window` is empolyed which slides from bottom of the image and keeps searching for line pixes until the top of the image is reached. This is shown in the figure below with the best line of fit for those pixel values which also depict the lane lines.

![alt text][image11]

<p style="text-align: center;"> Figure 7: Sliding Window Technique </p>

Once the x values of the image lines are found they are saved after each frame in the `Line()` class also defined in the `image_generator.py` file. These values are helpful for the next search for lane lines since now having these values from previous frames can be used to search around those locations inside a window area and speed up the process of searching using `not_blind_search` function. This leverages from the previous frames information so that we don't start searching for lane lines in the next frame from scratch.


#### 5. Calculating Radius of Curvature of the lane and the Vehicle Position with respect to lane center

Radius of curvature is calculated using the `rad_of_curvature` function in `image_generator.py` file. This is a nice [link](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) which explains the calculations. In our case, the f'(y) = 2Ay + B and f''(y) = 2A where A & B are the polynomial coefficients found while fitting a second order polyomial f(y) = Ay^2 + By + C.
The position of the vehicle with respect to the center of the lane is calculated in the `print_road_info` function on line 368 in `image_generator.py` file. Both radius of curvature and vehicle position are printed out on the image frames using the `print_road_info` function which gets displayed while the video is running.

#### 6. Recasting the detected lane on the original frames of the video

I implemented this step in lines 81 through 86 in my code in `main.py`. In `image_generator.py` file the `draw_lines` function takes the calculated matrices M and Minv from perspective transformation and warp back to original image. Here is an example of my result with the road information printed on the image as well:

![alt text][image12]

<p style="text-align: center;"> Figure 8: Warping back to original image and recasting lanes and road information </p>

---

### Pipeline (video)

#### 1. Results on the project video

Here's a link to my [video](https://youtu.be/RDlLg0vW2rM) showing final results.

---

### Discussion

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing my pipeline on single frames.  Although this program worked on the `project_video.mp`, more improvements on it like better thresholding techniques with color and gradients can make it work on the optional challenge videos as well which have frequent shadows and sharper turns.