#include <opencv2/opencv.hpp>
#include "VideoProcessingTrackingWithOpticFlow.h"

int main()
{
	auto src = cv::VideoCapture(0);
	VideoProcessingTrackingWithOpticFlow prc(src);
	prc.runLoops();
}
