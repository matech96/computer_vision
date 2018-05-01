#include "VideoProcessing.h"


VideoProcessing::VideoProcessing(cv::VideoCapture& src) : src(src){}

VideoProcessing::~VideoProcessing()
{
}

void VideoProcessing::runLoops()
{
	if (!src.isOpened())  // check if we succeeded
		return;

	cv::UMat edges;
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
	//auto localFrame = cv::UMat(frame);
	//cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);
	//cv::GaussianBlur(localFrame, localFrame, cv::Size(7, 7), 1.5, 1.5);
	//cv::Canny(localFrame, localFrame, 0, 30, 3);
	cv::imshow("frame", frame);
	if (cv::waitKey(30) >= 0)
	{
		return false;
	}
	return true;
}
