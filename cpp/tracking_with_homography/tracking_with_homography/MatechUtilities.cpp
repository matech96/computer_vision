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

cv::Mat MatechUtilities::getMaskAtCenter(const cv::UMat & frame)
{
	const int height = frame.rows;
	const int width = frame.cols;
	const cv::Scalar mask_ignore_color{ 0 };
	const cv::Scalar mask_accept_color{ 255 };
	const int recatangle_fill = -1;

	cv::Mat mask{ height, width, CV_8U, mask_ignore_color };
	const cv::Rect rectangle = MatechUtilities::getRectangleAtCenter(width, height);
	cv::rectangle(mask, rectangle, mask_accept_color, recatangle_fill);
	return mask;
}

std::vector<cv::Point2f> MatechUtilities::getPointsToTrack(const cv::UMat & frame, const cv::Mat & mask)
{
	const int max_count = 500;
	const double qualityLevel = 0.01;
	const double minDistance = 10;
	const int blockSize = 3;
	const int gradientSize = 3;
	std::vector<cv::Point2f> kpts{};
	cv::goodFeaturesToTrack(frame, kpts, max_count, qualityLevel, minDistance, mask, blockSize, gradientSize);
	return kpts;
}

std::vector<cv::Point2f> MatechUtilities::filterPoints(const std::vector<cv::Point2f>& points, const std::vector<uchar>& status)
{
	assert(points.size() == status.size());

	std::vector<cv::Point2f> res{};
	for (int i = 0; i < points.size(); i++)
	{
		if (status[i]) {
			res.push_back(points[i]);
		}
	}
	return res;
}
