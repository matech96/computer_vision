#include "TrackingCameraImage.h"

TrackingCameraImage::TrackingCameraImage(TrackingFrameProcessor &processor) : processor(processor)
{
	cv::namedWindow("frame", 1);
	auto src = cv::VideoCapture(0);
	VideoProcessing prc(src);
	prc.runLoops();
}

TrackingCameraImage::~TrackingCameraImage()
{
}
