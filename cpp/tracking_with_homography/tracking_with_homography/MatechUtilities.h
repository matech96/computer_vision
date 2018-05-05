#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace MatechUtilities
{
	std::vector<cv::Point> rectangleToPoints(cv::Rect &rectangle);
};

