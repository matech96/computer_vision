#pragma once

#include <opencv2\core.hpp>
class TrackingFrameProcessor
{
public:
	virtual bool startUpFcn(const cv::UMat &frame) = 0;
	virtual bool loopUpFcn(const cv::UMat &frame) = 0;
};

