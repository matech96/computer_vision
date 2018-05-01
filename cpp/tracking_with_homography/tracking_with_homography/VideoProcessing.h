#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
using loopFncT = bool(*)(const cv::UMat&);
class VideoProcessing
{
	cv::VideoCapture& src;
	loopFncT loopFnc;
public:
	VideoProcessing(cv::VideoCapture& src, loopFncT loopFnc);
	~VideoProcessing();

	void setLoopFnc(loopFncT l);
	void runLoops();
};

