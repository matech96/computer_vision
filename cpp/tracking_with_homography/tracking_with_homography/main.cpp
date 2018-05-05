#include <opencv2/opencv.hpp>
#include "TrackingFrameProcessorOpticFlow.h"

int main()
{
	auto src = cv::VideoCapture(0);
	TrackingFrameProcessorOpticFlow prc(src);
	prc.runLoops();
	cv::waitKey();
}
