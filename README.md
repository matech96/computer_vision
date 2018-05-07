# Computer vision
In this repository you can find notebooks of applications of computer vision algorithums using python or C++ with OpenCV.
## Tracking
[tracking.ipynb](python/tracking.ipynb) implements a histogramm based approach with CAMSHIFT. 
1. First we select a patch of the image. 
2. Use one or more changells to creat a histogramm: Check each color and put in the appropriate bin.
3. Store the histogramm.
4. For each image of the video feed:  
  > 1. Get the color at each location.
  > 2. Find the appropriate bucket.
  > 3. The value of the bucket is the weight of the pixel.
  > 4. Use mean shift to find the center.
 
[Results](https://www.youtube.com/playlist?list=PLrQlWh70z5dLRcFmsxvW5DjShTdsIha3-)
### OpticFlow
[tracking_with_homography C++ project](cpp/tracking_with_homography/tracking_with_homography/main.cpp) Findes keypoints in the selected region and tracks them with optic flow. Based on the motion of the points computes a homography and transforms the tracking window. However the computation of the homography has an unkown problem. The window is shifted to the upper left corner.
[TrackingObjectFlow.py](python/TrackingObjectFlow.py) Its a tweaked version of the above in python. It adds feature matching and resets the tracking window, if match is possible.
[Result](https://youtu.be/JtQz6ESbI6M)

