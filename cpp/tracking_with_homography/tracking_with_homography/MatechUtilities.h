#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace MatechUtilities
{
	std::vector<cv::Point> rectangleToPoints(const cv::Rect & rectangle);
	cv::Rect getRectangleAtCenter(const int width, const int height);
	cv::Mat getMaskAtCenter(const cv::UMat &frame);
};

