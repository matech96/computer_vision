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
	if (MatechUtilities::isButtonPushed())
	{
		setUpTrackingParameters(frame);
		return false;
	}
	return true;
}


void TrackingFrameProcessorOpticFlow::setUpTrackingParameters(const cv::UMat & frame)
{
	cv::Mat mask = MatechUtilities::getMaskAtCenter(frame);
	cv::UMat grayFrame = frame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
	prevKpts = MatechUtilities::getPointsToTrack(grayFrame, mask);
	prevFrame = grayFrame;
}


bool TrackingFrameProcessorOpticFlow::tracking(const cv::UMat & frame)
{
	cv::UMat grayFrame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

	auto tupel = MatechUtilities::trackPoints(prevFrame, grayFrame, prevKpts);
	cv::Mat H = std::get<0>(tupel);
	std::vector<cv::Point2f> filteredKpts = std::get<1>(tupel);

	cv::UMat localFrame = frame;
	DrawingUtilities::drawPointsOnto(localFrame, filteredKpts);
	try {
		cv::Mat vectors = MatechUtilities::pointsToHomogeneousMatrix(cornerPoints);
		cv::Mat res = H * vectors;
		cornerPoints = MatechUtilities::homogeneousMatrixToPoints(res);
		//Displaying
		DrawingUtilities::drawPolyShapeOnto(localFrame, cornerPoints);
		cv::imshow("frame", localFrame);
		// Updating
		cv::swap(prevFrame, grayFrame);
		std::swap(prevKpts, filteredKpts);
	}
	catch (const std::exception &exc) {
		std::cout << exc.what() << std::endl;
		return false;
	}
	if (MatechUtilities::isButtonPushed())
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
