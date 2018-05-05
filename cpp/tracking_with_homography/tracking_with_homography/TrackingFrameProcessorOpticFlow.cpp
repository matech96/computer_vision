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
		initializeCornerPoints(frame);
		state = TrackingState::BOX_DISPLAYING;
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
	auto localFrame = cv::UMat(frame);
	DrawingUtilities::drawPolyShapeOnto(localFrame, cornerPoints);
	cv::imshow("frame", localFrame);
	if (cv::waitKey(30) >= 0)
	{
		cv::Mat mask = MatechUtilities::getMaskAtCenter(localFrame);
		cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);

		prevKpts = MatechUtilities::getPointsToTrack(localFrame, mask);
		prevFrame = localFrame;

		return false;
	}
	return true;
}




bool TrackingFrameProcessorOpticFlow::tracking(const cv::UMat & frame)
{
	cv::UMat grayFrame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

	std::vector<cv::Point2f> kpts;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::calcOpticalFlowPyrLK(prevFrame, grayFrame, prevKpts, kpts, status, err);

	std::vector<cv::Point2f> filteredPrevKpts = MatechUtilities::filterPoints(prevKpts, status);
	std::vector<cv::Point2f> filteredKpts = MatechUtilities::filterPoints(kpts, status);
	cv::UMat localFrame = frame;
	for (cv::Point2f point : filteredKpts)
	{
		cv::circle(localFrame, point, 3, cv::Scalar(0, 255, 0), -1, 8);
	}
	try {
		cv::Mat H = cv::findHomography(filteredPrevKpts, filteredKpts);
		cv::Mat vectors = MatechUtilities::pointsToHomogeneousMatrix(cornerPoints);
		cv::Mat res = H * vectors;
		cornerPoints = MatechUtilities::homogeneousMatrixToPoints(res);
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
