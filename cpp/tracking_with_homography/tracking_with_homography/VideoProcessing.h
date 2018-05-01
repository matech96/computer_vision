#pragma once

#include <opencv2/opencv.hpp>
#include "MeassureTime.h"

class VideoProcessing
{
	cv::VideoCapture& src;
	bool runOneLoop();
	cv::UMat readFrameFormCamera();
	virtual bool processFrame(const cv::UMat &frame);
public:
	VideoProcessing(cv::VideoCapture& src);
	~VideoProcessing();

	void runLoops();
};

