#include "DrawingUtilities.h"

void DrawingUtilities::drawPolyShapeOnto(cv::UMat & frame, std::vector<cv::Point> cornerPoints)
{
	const cv::Scalar color = { 0, 255, 0 };
	int thickness = 10;
	cv::polylines(frame, cornerPoints, true, color, thickness);
}
