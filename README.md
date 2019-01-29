## Advanced Lane Finding

---

**Advanced Lane Finding Project**

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

[image0]: ./output_images/chessboard_corners.jpg "Checkboard corners"
[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_test1.png "Road Transformed"
[image3]: ./output_images/thresholding.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/fit_polyline.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./output_images/output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 15 through 31 of the file `lane_lines_detection.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The following is the result of chessboard corner detection:

![chessboard_corner][image0]  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistort][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![undistored_test1][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The functions for thresholding are defined through line #51 to line #61 and the steps to apply the thresholding are from line #179 to line #185 in `lane_lines_detection.py`.
I found out that when working on the grayscale image, the yellow line is easily mixed with the background while s channel can solve the problem. 
But using s channel alone will omit some white lines. So I used the combination of the grayscale image and the s channel. Here's an example of my output for this step:

![thresholding_image][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_to_bird_eye()`, which appears in lines 35 through 48 in the file `lane_lines_detection.py`.  The `warp_to_bird_eye()` function takes as inputs an image (`img`). I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[571, 468],
    [244.661, 693.952],
    [1042.08, 681.048],
    [716, 468]])
dst = np.float32(
    [[img.shape[1] / 4, 0],
    [img.shape[1] / 4, img.shape[0]],
    [img.shape[1] * 3 / 4, img.shape[0]],
    [img.shape[1] * 3 / 4, 0]])
```

This resulted in the following source and destination points:

| Source            | Destination   | 
|:-----------------:|:-------------:| 
| 571, 468          | 320, 0        | 
| 244.661, 693.952  | 320, 720      |
| 1042.08, 681.048  | 960, 720      |
| 716, 468          | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

I found the lane line pixels by first calculate the histogram peak of the bottom half images and look for the pixels by iterating the windows. This is for the first frame.
For the following frame, I found the lane line pixels by finding the pixels around the previous found polylines. Then, the final polyline is obtained by averaging the 5 previous found polylines.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 204 through 205 in my code in `lane_lines_detection.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 191 through 212 in my code in `lane_lines_detection.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The techniques that I used include applying sobel on the image and get the binary images by combining thresholding the magnitude and the color image.
Also I apply the window searching to find the lane line pixels and fit the pixels with polylines.

The problem I encountered is that the parameters don't work well when the lighting condition is changed. Also, when the lane line curve is too curved,
the algorithm will also fail. Third, if there are some outliers on the lane line, for example, pedestrian, the algorithm can't distinguish lane line
the obstacles so it will also fail. The further improvement I'm thinking is to apply semantic segmentation on the image first to extract the lane line
then apply our pipeline. We could use deep learning to perform the segmentation on the image and then detect the lane line.