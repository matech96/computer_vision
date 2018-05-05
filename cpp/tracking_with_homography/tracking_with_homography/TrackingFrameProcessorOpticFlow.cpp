#include "TrackingFrameProcessorOpticFlow.h"
#include "DrawingUtilities.h"


void TrackingFrameProcessorOpticFlow::initializeCornerPoints(const cv::UMat & frame)
{
	const int height = frame.rows;
	const int width = frame.cols;
	const cv::Rect rectangle = MatechUtilities::getRectangleAtCenter(width, height);
	cornerPoints = MatechUtilities::rectangleToPoints(rectangle);
}

bool TrackingFrameProcessorOpticFlow::processFrame(const cv::UMat & frame)
{
	switch (state) {
	case TrackingState::INITIALIZATION:
	{
	}
	break;
	case TrackingState::BOX_DISPLAYING:
	{
		bool do_continue = displayBox(frame);
		if (!do_continue) {
			std::cout << "Tracking" << std::endl;
			state = TrackingState::TRACKING;
		}
	}
	break;
	case TrackingState::TRACKING:
	{
		return tracking(frame);
	}
	default:
		return false;
	}
	return true;
}

bool TrackingFrameProcessorOpticFlow::displayBox(const cv::UMat & frame)
{
	initializeCornerPoints(frame);
	auto localFrame = cv::UMat(frame);
	DrawingUtilities::drawPolyShapeOnto(localFrame, cornerPoints);
	cv::imshow("frame", localFrame);
	if (cv::waitKey(30) >= 0)
	{
		int max_count = 500;
		double qualityLevel = 0.01;
		double minDistance = 10;
		cv::Mat mask = MatechUtilities::getMaskAtCenter(localFrame);
		cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);
		cv::goodFeaturesToTrack(localFrame, prevKpts, max_count, qualityLevel, minDistance, mask, 3, 3, 0, 0.04);
		prevFrame = localFrame;
		cv::drawKeypoints(frame, keypoints, localFrame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		cv::imshow("frame", localFrame);
		return false;
	}
	return true;
}


bool TrackingFrameProcessorOpticFlow::tracking(const cv::UMat & frame)
{
	cv::UMat localFrame;
	frame.copyTo(localFrame);
	cv::UMat grayFrame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
	std::vector<cv::Point2f> kpts;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Size winSize = { 31, 31 };
	cv::TermCriteria termcrit = { cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03 };
	cv::calcOpticalFlowPyrLK(prevFrame, grayFrame, prevKpts, kpts, status, err);// , winSize, 3, termcrit, 0, 0.001);
	int i = 0;
	std::vector<cv::Point2f> filteredPrevKpts = {};
	std::vector<cv::Point2f> filteredKpts = {};
	for (cv::Point2f point : kpts)
	{
		if (!status[i]) {
			continue;
		}
		filteredPrevKpts.push_back(prevKpts[i]);
		filteredKpts.push_back(kpts[i]);
		cv::circle(localFrame, point, 3, cv::Scalar(0, 255, 0), -1, 8);
		i++;
	}
	try {
		cv::Mat H = cv::findHomography(filteredPrevKpts, filteredKpts);
		cv::Mat vectors(3, 4, CV_64F, 1);
		for (size_t i = 0; i < 4; i++)
		{
			vectors.at<double>(0, i) = cornerPoints[i].x;
			vectors.at<double>(1, i) = cornerPoints[i].y;
		}
		cv::Mat res = H * vectors;
		for (size_t i = 0; i < 4; i++)
		{
			double h = res.at<double>(2, i);
			cornerPoints[i].x = res.at<double>(0, i) / h;
			cornerPoints[i].y = res.at<double>(1, i) / h;
		}
		cv::polylines(localFrame, cornerPoints, true, cv::Scalar(255, 255, 0));
		cv::imshow("frame", localFrame);
		cv::swap(prevFrame, grayFrame);
		prevKpts = filteredKpts;
	}
	catch (const std::exception &exc) {
		std::cout << exc.what() << std::endl;
		return false;
	}
	if (cv::waitKey(30) >= 0)
	{
		return false;
	}
	return true;
}

TrackingFrameProcessorOpticFlow::TrackingFrameProcessorOpticFlow(cv::VideoCapture & src) : VideoProcessing(src)
{
}

TrackingFrameProcessorOpticFlow::~TrackingFrameProcessorOpticFlow()
{
}
