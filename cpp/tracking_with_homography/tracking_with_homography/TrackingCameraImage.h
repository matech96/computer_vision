#pragma once
#include "TrackingFrameProcessor.h"
#include <opencv2/opencv.hpp>
#include "VideoProcessing.h"
class TrackingCameraImage
{
	TrackingFrameProcessor &processor;
public:
	TrackingCameraImage(TrackingFrameProcessor& processor);
	~TrackingCameraImage();
};

