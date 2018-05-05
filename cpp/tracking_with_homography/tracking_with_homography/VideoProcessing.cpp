#include "VideoProcessing.h"
#include "MatechUtilities.h"


VideoProcessing::VideoProcessing(cv::VideoCapture& src) : src(src){}

VideoProcessing::~VideoProcessing()
{
}

void VideoProcessing::runLoops()
{
	if (!src.isOpened())  // check if we succeeded
		return;

	while (true)
	{
		bool do_continue = runOneLoop();
		if (!do_continue) {
			break;
		}
	}
}

bool VideoProcessing::runOneLoop()
{
	MeassureTime t = MeassureTime();
	cv::UMat frame = readFrameFormCamera();
	return processFrame(frame);
}

cv::UMat VideoProcessing::readFrameFormCamera()
{
	cv::UMat frame;
	src >> frame; // get a new frame from camera
	const int flipCodeYAxis = 1;
	cv::flip(frame, frame, flipCodeYAxis);
	return frame;
}

bool VideoProcessing::processFrame(const cv::UMat & frame)
{
	cv::imshow("frame", frame);
	if (MatechUtilities::isButtonPushed())
	{
		return false;
	}
	return true;
}
