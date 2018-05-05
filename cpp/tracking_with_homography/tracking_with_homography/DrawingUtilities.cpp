#include "DrawingUtilities.h"

void DrawingUtilities::drawPolyShapeOnto(cv::UMat & frame, const std::vector<cv::Point> & cornerPoints)
{
	const cv::Scalar color = { 0, 255, 0 };
	int thickness = 10;
	cv::polylines(frame, cornerPoints, true, color, thickness);
}

void DrawingUtilities::drawPointsOnto(cv::UMat & frame, const std::vector<cv::Point2f>& points)
{
	for (const cv::Point2f &point : points)
	{
		cv::circle(frame, point, 3, cv::Scalar(255, 0, 0), -1, 8);
	}
}
