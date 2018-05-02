#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "VideoProcessing.h"

enum class TrackingState { BOX_DISPLAYING, TRACKING };

class TrackingFrameProcessorOpticFlow : public VideoProcessing
{
	TrackingState state = { TrackingState::BOX_DISPLAYING};
	std::vector<cv::KeyPoint> keypoints = {};
	cv::UMat descriptors = {};
	std::vector<cv::Point> corners = {};

	cv::UMat prevFrame = {};
	std::vector<cv::Point2f> prevKpts = {};

	bool processFrame(const cv::UMat &frame) override;
	bool displayBox(const cv::UMat & frame);
public:
	TrackingFrameProcessorOpticFlow(cv::VideoCapture& src);
	~TrackingFrameProcessorOpticFlow();
};

