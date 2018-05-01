#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
//#include<conio.h>           // may have to modify this line if not using Windows
#include "VideoProcessing.h"
#include "TrackingCameraImage.h"

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

bool setUpFnc(const cv::UMat &frame) {
	int height = frame.rows;
	int width = frame.cols;
	auto localFrame = cv::UMat(frame);
	int x = width / 4;
	int y = height / 4;
	int w = width / 2;
	int h = height / 2;
	std::vector<cv::Point> corners = {
			cv::Point(x, y),
			cv::Point(x+w, y),
			cv::Point(x+w, y+h),
			cv::Point(x, y+h)
	};
	cv::polylines(localFrame, corners, true, cv::Scalar(0, 255, 0));
	cv::imshow("frame", localFrame);
	if (cv::waitKey(30) >= 0)
	{
		return false;
	}
	return true;
}

int main(int, char**)
{
	TrackingCameraImage(TrackingOpticFlow());
}
