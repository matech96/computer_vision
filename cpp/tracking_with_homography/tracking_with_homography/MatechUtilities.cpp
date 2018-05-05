#include "MatechUtilities.h"

std::vector<cv::Point> MatechUtilities::rectangleToPoints(cv::Rect & rectangle)
{
	int x = rectangle.x;
	int y = rectangle.y;
	int w = rectangle.width;
	int h = rectangle.height;
	return {
		cv::Point(x, y),
		cv::Point(x + w, y),
		cv::Point(x + w, y + h),
		cv::Point(x, y + h)
	};
}
