#include "TrackingFrameProcessorOpticFlow.h"


bool TrackingFrameProcessorOpticFlow::processFrame(const cv::UMat & frame)
{
	switch (state) {
	case TrackingState::BOX_DISPLAYING:
	{
		bool do_continue = displayBox(frame);
		if (!do_continue) {
			state = TrackingState::TRACKING;
			cv::UMat localFrame;
			cv::drawKeypoints(frame, keypoints, localFrame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
			cv::imshow("frame", localFrame);
		}
		break; 
	}
	case TrackingState::TRACKING:
	{
		if (cv::waitKey(30) >= 0)
		{
			return false;
		}
		break;
	}
	default:
		return false;
	}
	return true;
}

bool TrackingFrameProcessorOpticFlow::displayBox(const cv::UMat & frame)
{

	int height = frame.rows;
	int width = frame.cols;
	auto localFrame = cv::UMat(frame);
	int x = width / 4;
	int y = height / 4;
	int w = width / 2;
	int h = height / 2;
	const cv::Scalar color(0, 255, 0);
	std::vector<cv::Point> corners = {
		cv::Point(x, y),
		cv::Point(x + w, y),
		cv::Point(x + w, y + h),
		cv::Point(x, y + h)
	};
	cv::polylines(localFrame, corners, true, color);
	cv::imshow("frame", localFrame);
	if (cv::waitKey(30) >= 0)
	{
		cv::Ptr<cv::ORB> detector = cv::ORB::create();
		cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);
		cv::Mat mask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
		cv::rectangle(mask, cv::Rect(x, y, w, h), cv::Scalar(255), -1);
		//cv::imshow("frame", mask);
		//std::cout << static_cast<int>(mask.at<uchar>(x - 5, y - 5)) << std::endl;
		//std::cout << static_cast<int>(mask.at<uchar>(x + 5, y + 5)) << std::endl;
		//cv::waitKey(30);
		detector->detectAndCompute(localFrame, mask, keypoints, descriptors);
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
