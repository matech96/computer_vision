#include "TrackingFrameProcessorOpticFlow.h"
#include "EnumerateObject.h"


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
		cv::UMat localFrame;
		cv::cvtColor(frame, localFrame, cv::COLOR_BGR2GRAY);
		std::vector<cv::Point2f> kpts;
		std::vector<uchar> status;
		std::vector<float> err;
		cv::Size winSize = { 31, 31 };
		cv::TermCriteria termcrit = { cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03 };
		cv::calcOpticalFlowPyrLK(prevFrame, localFrame, prevKpts, kpts, status, err, winSize, 3, termcrit, 0, 0.001);
		int i = 0;
		for (cv::Point2f point : kpts)
		{
			if (!status[i]) {
				continue;
			}
			cv::circle(localFrame, point, 3, cv::Scalar(0, 255, 0), -1, 8);
			i++;
		}
		cv::imshow("frame", localFrame);
		cv::swap(localFrame, prevFrame);
		std::swap(kpts, prevKpts);
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
		cv::cvtColor(localFrame, localFrame, cv::COLOR_BGR2GRAY);
		int getUnlimitedCorners = 500;
		double qualityLevel = 0.01;
		double minDistance = 5;
		cv::Mat mask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
		cv::rectangle(mask, cv::Rect(x, y, w, h), cv::Scalar(255), -1);
		cv::goodFeaturesToTrack(localFrame, prevKpts, getUnlimitedCorners, qualityLevel, minDistance, mask, 3, 3, 0, 0.04);
		cv::swap(localFrame, prevFrame);
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
