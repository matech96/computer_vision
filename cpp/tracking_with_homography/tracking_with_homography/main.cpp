#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
//#include<conio.h>           // may have to modify this line if not using Windows
#include "VideoProcessing.h"

bool edgeDetectionLoopFnc(const cv::UMat &frame)
{
	auto localFrame = cv::UMat(frame);
	cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(localFrame, localFrame, cv::Size(7, 7), 1.5, 1.5);
	cv::Canny(localFrame, localFrame, 0, 30, 3);
	cv::imshow("frame", localFrame);
	if (cv::waitKey(30) >= 0) 
	{
		return false;
	}
	return true;
}

int main(int, char**)
{
	cv::namedWindow("frame", 1);
	auto src = cv::VideoCapture(0);
	VideoProcessing prc(src, edgeDetectionLoopFnc);
	prc.runLoops();
}
