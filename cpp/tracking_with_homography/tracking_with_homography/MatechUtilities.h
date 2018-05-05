#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace MatechUtilities
{
	std::vector<cv::Point> rectangleToPoints(const cv::Rect & rectangle);
	cv::Rect getRectangleAtCenter(const int width, const int height);
	cv::Mat getMaskAtCenter(const cv::UMat &frame);
	std::vector<cv::Point2f> getPointsToTrack(const cv::UMat &frame, const cv::Mat &mask);
	std::vector<cv::Point2f> filterPoints(const std::vector<cv::Point2f> &points, const std::vector<uchar> &status);
	cv::Mat pointsToHomogeneousMatrix(const std::vector<cv::Point2i>& points);
	std::vector<cv::Point2i> homogeneousMatrixToPoints(const cv::Mat &points);
	bool isButtonPushed();
	std::tuple<cv::Mat, std::vector<cv::Point2f>> trackPoints(const cv::UMat & prevGeryFrame, const cv::UMat & geryFrame, const std::vector<cv::Point2f> &prevPoints);
	std::vector<cv::Point> transformPointsWithHomography(const cv::Mat &H, const std::vector<cv::Point> &points);
};

