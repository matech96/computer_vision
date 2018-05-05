#pragma once
#include <opencv2/opencv.hpp>
namespace DrawingUtilities
{
	void drawPolyShapeOnto(cv::UMat & frame, std::vector<cv::Point> cornerPoints);
};

