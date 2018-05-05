#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "VideoProcessing.h"
#include "MatechUtilities.h"

enum class TrackingState { INITIALIZATION, BOX_DISPLAYING, TRACKING };

class VideoProcessingTrackingWithOpticFlow : public VideoProcessing
{
	TrackingState state = { TrackingState::INITIALIZATION };
	std::vector<cv::Point2i> cornerPoints = {};

	cv::UMat prevFrame = {};
	std::vector<cv::Point2f> prevKpts = {};

	void initializeCornerPoints(const cv::UMat & frame);
	bool processFrame(const cv::UMat &frame) override;
	bool tracking(const cv::UMat & frame);
	void displayTrackingState(cv::UMat &localFrame, std::vector<cv::Point2f> &filteredKpts);
	bool displayBox(const cv::UMat & frame);
	void setUpTrackingParameters(const cv::UMat & localFrame);
public:
	VideoProcessingTrackingWithOpticFlow(cv::VideoCapture& src);
	~VideoProcessingTrackingWithOpticFlow();
};
