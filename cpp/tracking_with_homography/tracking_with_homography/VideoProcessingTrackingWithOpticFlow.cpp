#include "VideoProcessingTrackingWithOpticFlow.h"
#include "DrawingUtilities.h"


void VideoProcessingTrackingWithOpticFlow::initializeCornerPoints(const cv::UMat & frame)
{
	const int height = frame.rows;
	const int width = frame.cols;
	const cv::Rect rectangle = MatechUtilities::getRectangleAtCenter(width, height);
	cornerPoints = MatechUtilities::rectangleToPoints(rectangle);
}

bool VideoProcessingTrackingWithOpticFlow::processFrame(const cv::UMat & frame)
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

bool VideoProcessingTrackingWithOpticFlow::displayBox(const cv::UMat & frame)
{
	auto localFrame = cv::UMat(frame);
	DrawingUtilities::drawPolyShapeOnto(localFrame, cornerPoints);
	cv::imshow("frame", localFrame);
	if (MatechUtilities::isButtonPushed())
	{
		setUpTrackingParameters(frame);
		return false;
	}
	return true;
}


void VideoProcessingTrackingWithOpticFlow::setUpTrackingParameters(const cv::UMat & frame)
{
	cv::Mat mask = MatechUtilities::getMaskAtCenter(frame);
	cv::UMat grayFrame = frame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
	prevKpts = MatechUtilities::getPointsToTrack(grayFrame, mask);
	prevFrame = grayFrame;
}


bool VideoProcessingTrackingWithOpticFlow::tracking(const cv::UMat & frame)
{
	cv::UMat localFrame = frame;
	cv::UMat grayFrame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

	auto tupel = MatechUtilities::trackPoints(prevFrame, grayFrame, prevKpts);
	cv::Mat H = std::get<0>(tupel);
	std::vector<cv::Point2f> filteredKpts = std::get<1>(tupel);
	cornerPoints = MatechUtilities::transformPointsWithHomography(H, cornerPoints);
	displayTrackingState(localFrame, filteredKpts);
	// Updating
	cv::swap(prevFrame, grayFrame);
	std::swap(prevKpts, filteredKpts);

	if (MatechUtilities::isButtonPushed())
	{
		return false;
	}
	return true;
}

void VideoProcessingTrackingWithOpticFlow::displayTrackingState(cv::UMat &localFrame, std::vector<cv::Point2f> &filteredKpts)
{

	DrawingUtilities::drawPointsOnto(localFrame, filteredKpts);
	DrawingUtilities::drawPolyShapeOnto(localFrame, cornerPoints);
	cv::imshow("frame", localFrame);
}

VideoProcessingTrackingWithOpticFlow::VideoProcessingTrackingWithOpticFlow(cv::VideoCapture & src) : VideoProcessing(src)
{
}

VideoProcessingTrackingWithOpticFlow::~VideoProcessingTrackingWithOpticFlow()
{
}
