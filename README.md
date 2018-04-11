# Computer vision
In this repository you can find notebooks of applications of computervision algorithum using python and OpenCV.
## Tracking
[tracking.ipynb](tracking.ipynb) implements a histogramm based approach with CAMSHIFT. 
1. First we select a patch of the image. 
2. Use one or more changells to creat a histogramm: Check each color and put in the appropriate bin.
4. Store the histogramm.
5. For each image of the video feed:
  > 1. Get the color at each location.
  > 2. Find the appropriate bucket.
  > 3. The value of the bucket is the weight of the pixel.
  > 4. Use mean shift to find the center.
[Results](https://www.youtube.com/playlist?list=PLrQlWh70z5dLRcFmsxvW5DjShTdsIha3-) 
