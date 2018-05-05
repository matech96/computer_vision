#include "MatechUtilities.h"

std::vector<cv::Point> MatechUtilities::rectangleToPoints(const cv::Rect & rectangle)
{
	const int x = rectangle.x;
	const int y = rectangle.y;
	const int w = rectangle.width;
	const int h = rectangle.height;
	return {
		cv::Point(x, y),
		cv::Point(x + w, y),
		cv::Point(x + w, y + h),
		cv::Point(x, y + h)
	};
}


cv::Rect MatechUtilities::getRectangleAtCenter(const int width, const int height)
{
	const int x = width / 4;
	const int y = height / 4;
	const int w = width / 2;
	const int h = height / 2;
	return cv::Rect(x, y, w, h);
}