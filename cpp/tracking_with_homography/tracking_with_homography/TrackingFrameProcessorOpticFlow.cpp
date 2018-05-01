#include "TrackingFrameProcessorOpticFlow.h"


bool TrackingFrameProcessorOpticFlow::processFrame(const cv::UMat & frame)
{
	int height = frame.rows;
	int width = frame.cols;
	auto localFrame = cv::UMat(frame);
	int x = width / 4;
	int y = height / 4;
	int w = width / 2;
	int h = height / 2;
	std::vector<cv::Point> corners = {
		cv::Point(x, y),
		cv::Point(x + w, y),
		cv::Point(x + w, y + h),
		cv::Point(x, y + h)
	};
	cv::polylines(localFrame, corners, true, cv::Scalar(0, 255, 0));
	cv::imshow("frame", localFrame);
	if (cv::waitKey(30) >= 0)
	{
		int minHessian = 400;
		cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
		cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);
		detector->detectAndCompute(localFrame, cv::UMat(), keypoints, descriptors);
		cv::drawKeypoints(frame, keypoints, localFrame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		return true;
	}
	return true;
}

TrackingFrameProcessorOpticFlow::TrackingFrameProcessorOpticFlow(cv::VideoCapture & src) : VideoProcessing(src)
{
}

TrackingFrameProcessorOpticFlow::~TrackingFrameProcessorOpticFlow()
{
}
