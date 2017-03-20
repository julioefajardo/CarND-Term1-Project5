## Vehicle Detection Project

**Self-Driving Cars Nanodegree - Project 5**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG.png
[image2]: ./output_images/HOG2.png
[image3]: ./output_images/SVM.png
[image4]: ./output_images/test3.png
[image5]: ./output_images/Final.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png


### Histogram of Oriented Gradients (HOG)
The histogram of oriented gradients (HOG) is a feature descriptor popularly used in computer vision, particularly on object detection field. This technique, basically counts occurrences of gradient orientation in localized portions of an image. The function `skimage.hog()` was used and tested in this stage using different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  Random vehicle and non-vehicle images were picked from data sets and then displayed in order to decided the best `skimage.hog()`  parameters for the project. The parameters were selected by experimentation and following the recommendations of this [blog](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906#.ew12hhpj9).

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(4, 4)` and `cells_per_block=(1, 1)`:

![alt text][image1]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

![alt text][image2]

The code for this step is contained in the third and fourth code cell of the IPython notebook named [Training_SVM](Training_SVM.ipynb). 


#### Support Vector Machine Classifier

The vehicle and non-vehicle datasets provided were explored with the aim of verifying if they are balanced. Udacity provided 8.792 images of vehicles and 8.968 images of non-vehicles of 64x64 pixels each, from  GTI vehicle image database and the KITTI vision benchmark suite.  

A Linear Support Vector Machine based on the `SVC` function from the `scikit-learn` machine learning package. HOG features with the following parameters (`color_space=YCrCb`, `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`) were chosen to train a SVM with linear kernel. With these settings a `0.9778` of accuracy was obtained. The feature extraction stage were performed using the `extract_features()` function provided on the lectures. The data was scaled and normalized using the `StandardScaler()` from `sklearn.preprocessing` package.  

The code for this step is contained in code cells from four to six of the IPython notebook named [Training_SVM](Training_SVM.ipynb). 

![alt text][image3]

### Sliding Window Search

A basic sliding window search was implemented using the `slide_window()` and `search_windows()` in order to detect the areas where vehicles supposed to be on each frame of the video (the `search_windows()` takes advantage of the pre trained SVM to detect vehicles). A window size of `96x96` pixels was chosen based on experimentation. The area of the sliding window search was limited to `x_start_stop=(200, 1180)` and `y_start_stop=(400, 700)` and the overlapping percentage was fixed on 0.7 based on experimentation. 

In order to avoid false positives, positive vehicle detections are recorded in each frame of the video and a heatmap was created and then thresholded to identify vehicle positions.  The `scipy.ndimage.measurements.label()` was used to identify individual blobs in the heatmap assuming that each blob corresponded to a vehicle.  Then, bounding boxes are drawn to cover the area of each blob detected. Also, a class named `HotWindows()`, a queue of lists of bounding boxes were taken from this [project](https://github.com/georgesung/vehicle_detection_hog_svm/blob/master/HotWindows.py).

The code for this step is contained in the third and seventh code cell of the IPython notebook named [P5](P5.ipynb). 

![alt text][image4]

An image of the final implementation is shown below:

![alt text][image5]

### Video Implementation
The code for this step is contained in the code cells 10 and 11 of the IPython notebook named [P5](P5.ipynb), on the function named `pipeline()`. 

#### A youtube video processed with the algorithm are shown below:

[![Alt text for your video](https://img.youtube.com/vi/SJmWCHr21C8/0.jpg)](https://www.youtube.com/watch?v=Y30C_FkGIHs)

---

### Discussion

It is been known the amazing power of the combination of computer vision and machine learning tools brings on robotics field. The pipeline implemented on the project video follows a series of obvious steps in order to detect vehicles on the road under controlled conditions, only windows of a fixed size are used to search the cars, affecting the performance because fails  detecting vehicles in some areas of the frame (e.g. vehicles farther away), however, some improvements can be done, by implementing a sliding window search with at least 3 different window sizes and then combining in order to obtain more accurate data on the heatmap to avoid false positives.  Proper implementation on different ways can achieve the goal for this project (e.g. Deep Learning approach), i decided to follow the methods that Udacity proposed on the lectures. However, issues can appear when many vehicles appear on the road. The pipeline can be improved by fine tuning of the methods described and by adding multiple sliding window search method as proposed before, also, I believe that I will continue working on the project, to improve the tracking of vehicles, and include motorcycles, pedestrians, bicycles and traffic signals in order to get a useful tool in an autonomous car.

