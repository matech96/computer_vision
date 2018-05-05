#pragma once
#include <opencv2/opencv.hpp>
namespace DrawingUtilities
{
	void drawPolyShapeOnto(cv::UMat & frame, const std::vector<cv::Point> & cornerPoints);
	void drawPointsOnto(cv::UMat & frame, const std::vector<cv::Point2f>& points);
};

