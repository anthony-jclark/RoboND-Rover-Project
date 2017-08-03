## Project: Search and Sample Return

---


**The goals / steps of this project are the following:**

**Training / Calibration**

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
<!-- ### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. -->

---
### Writeup / README

<!-- #### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. -->

<!-- Writeup provided in markdown format on Github. -->

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

<!-- Describe in your writeup (and identify where in your code) how you modified or added functions to add obstacle and rock sample identification. -->

Using the calibration image (`'../calibration_images/example_rock1.jpg'`) as a test, I altered the `color_thresh` function so that it would indicate the presence of a rock sample. Specifically, I took the suggested method of adding a `rbg_thresh_max` parameter such that color selection could occur within a specified range (comments removed for brevity):

```python
def color_thresh(img, rgb_thresh_min=(160, 160, 160), rgb_thresh_max=(255, 255, 255)):
    color_select = np.zeros_like(img[:,:,0])
    thresh_select = (rgb_thresh_min[0] <= img[:,:,0]) & (img[:,:,0] <= rgb_thresh_max[0]) \
                  & (rgb_thresh_min[1] <= img[:,:,1]) & (img[:,:,1] <= rgb_thresh_max[1]) \
                  & (rgb_thresh_min[2] <= img[:,:,2]) & (img[:,:,2] <= rgb_thresh_max[2])
    color_select[thresh_select] = 1
    return color_select
```

I used the default values for detecting navigable terrain and I tried two methods for selecting obstacles. Finally, I used an RGB chart to empirically select thresholds for the rock samples.

```python
navigable_terrain = color_thresh(warped)
rock_samples = color_thresh(warped, rgb_thresh_min=(100, 100, 0), rgb_thresh_max=(255,255,80))

# Obstacles attempt 1
obstacles = -(navigable_terrain - 1)

# Obstacles attempt 2
obstacles = color_thresh(warped, rgb_thresh_min=(0, 0, 0), rgb_thresh_max=(120, 120, 120))
```

In the first attempt, I simply used the negation of the `navigable_terrain` variable. However, I noticed that this had a negative impact on map fidelity. Attempt 2 does a better job of finding "definite" obstacles. Note: this does mean that some parts of the image remain unclassified. The image below shows the images generated with regards to the sample rock image.

![Image Classification](./writeup/image_classification.png)


#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.
And another!

<!-- Describe in your writeup how you modified the process_image() to demonstrate your analysis and how you created a worldmap. Include your video output with your submission. -->

![alt text][image2]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.



![alt text][image3]


