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
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		cv::UMat frame;
		src >> frame; // get a new frame from camera
		const int flipCodeYAxis = 1;
		cv::flip(frame, frame, flipCodeYAxis);

		//cv::GaussianBlur(frame, frame, cv::Size(7, 7), 1.5, 1.5);
		//cv::Canny(frame, frame, 0, 30, 3);
		//cv::imshow("frame", frame);
		bool do_continue = loopFnc(frame);
		if (!do_continue) {
			break;
		}

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		std::cout << (1e6 / duration) << std::endl;
	}
}
