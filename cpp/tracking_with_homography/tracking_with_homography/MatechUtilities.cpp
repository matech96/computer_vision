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

cv::Mat MatechUtilities::pointsToHomogeneousMatrix(const std::vector<cv::Point2i>& points)
{
	cv::Mat res(3, 4, CV_64F, 1);
	for (size_t i = 0; i < 4; i++)
	{
		res.at<double>(0, i) = static_cast<double>(points[i].x);
		res.at<double>(1, i) = static_cast<double>(points[i].y);
	}
	return res;
}

std::vector<cv::Point2i> MatechUtilities::homogeneousMatrixToPoints(const cv::Mat & matrix)
{
	assert(matrix.rows == 3);
	std::vector<cv::Point2i> res{};
	for (size_t i = 0; i < matrix.cols; i++)
	{
		double h = matrix.at<double>(2, i);
		const int x = matrix.at<double>(0, i) / h;
		const int y = matrix.at<double>(1, i) / h;
		res.push_back({ x, y });
	}
	return res;
}

bool MatechUtilities::isButtonPushed()
{
	return cv::waitKey(30) >= 0;
}

std::tuple<cv::Mat, std::vector<cv::Point2f>> MatechUtilities::trackPoints(const cv::UMat & prevGeryFrame, const cv::UMat & geryFrame, const std::vector<cv::Point2f>& prevPoints)
{
	std::vector<cv::Point2f> points;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::calcOpticalFlowPyrLK(prevGeryFrame, geryFrame, prevPoints, points, status, err);

	std::vector<cv::Point2f> filteredPrevPoints = MatechUtilities::filterPoints(prevPoints, status);
	std::vector<cv::Point2f> filteredPoints = MatechUtilities::filterPoints(points, status);
	cv::Mat H = cv::findHomography(filteredPrevPoints, filteredPoints);
	return { H, filteredPoints };
}

std::vector<cv::Point> MatechUtilities::transformPointsWithHomography(const cv::Mat & H, const std::vector<cv::Point>& points)
{
	const cv::Mat vectors = MatechUtilities::pointsToHomogeneousMatrix(points);
	const cv::Mat res = H * vectors;
	return MatechUtilities::homogeneousMatrixToPoints(res);
}

