#include "VideoProcessing.h"


VideoProcessing::VideoProcessing(cv::VideoCapture& src, loopFncT loopFnc) : src(src), loopFnc(loopFnc){}

VideoProcessing::~VideoProcessing()
{
}

void VideoProcessing::setLoopFnc(loopFncT l)
{
	loopFnc = l;
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
	bool do_continue = loopFnc(frame);

	return do_continue;
}

cv::UMat VideoProcessing::readFrameFormCamera()
{
	cv::UMat frame;
	src >> frame; // get a new frame from camera
	const int flipCodeYAxis = 1;
	cv::flip(frame, frame, flipCodeYAxis);
	return frame;
}
