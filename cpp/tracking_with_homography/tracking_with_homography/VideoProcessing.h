#pragma once

#include <opencv2/opencv.hpp>
#include "MeassureTime.h"
using loopFncT = bool(*)(const cv::UMat&);
class VideoProcessing
{
	cv::VideoCapture& src;
	loopFncT loopFnc;
	bool runOneLoop();
	cv::UMat readFrameFormCamera();
public:
	VideoProcessing(cv::VideoCapture& src, loopFncT loopFnc);
	~VideoProcessing();

	void setLoopFnc(loopFncT l);
	void runLoops();
};

