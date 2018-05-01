#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "VideoProcessing.h"
#include "TrackingFrameProcessor.h"

class TrackingFrameProcessorOpticFlow : VideoProcessing
{
	std::vector<cv::KeyPoint> keypoints;
	cv::UMat descriptors;
	bool processFrame(const cv::UMat &frame) override;
public:
	TrackingFrameProcessorOpticFlow(cv::VideoCapture& src);
	~TrackingFrameProcessorOpticFlow();
};

