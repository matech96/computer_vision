#include "TrackingFrameProcessorOpticFlow.h"
#include "EnumerateObject.h"


bool TrackingFrameProcessorOpticFlow::processFrame(const cv::UMat & frame)
{
	switch (state) {
	case TrackingState::BOX_DISPLAYING:
	{
		bool do_continue = displayBox(frame);
		if (!do_continue) {
			cv::UMat localFrame;
			cv::drawKeypoints(frame, keypoints, localFrame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
			cv::imshow("frame", localFrame);
			state = TrackingState::TRACKING;
		}
		break; 
	}
	case TrackingState::TRACKING:
	{
		cv::UMat localFrame;
		frame.copyTo(localFrame);
		cv::UMat grayFrame;
		cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
		std::vector<cv::Point2f> kpts;
		std::vector<uchar> status;
		std::vector<float> err;
		cv::Size winSize = { 31, 31 };
		cv::TermCriteria termcrit = { cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03 };
		cv::calcOpticalFlowPyrLK(prevFrame, grayFrame, prevKpts, kpts, status, err);// , winSize, 3, termcrit, 0, 0.001);
		int i = 0;
		std::vector<cv::Point2f> filteredPrevKpts = {};
		std::vector<cv::Point2f> filteredKpts = {};
		for (cv::Point2f point : kpts)
		{
			if (!status[i]) {
				continue;
			}
			filteredPrevKpts.push_back(prevKpts[i]);
			filteredKpts.push_back(kpts[i]);
			cv::circle(localFrame, point, 3, cv::Scalar(0, 255, 0), -1, 8);
			i++;
		}
		cv::Mat H = cv::findHomography(prevKpts, filteredKpts);
		cv::Mat vectors(3, 4, CV_64F, 1);
		for (size_t i = 0; i < 4; i++)
		{
			vectors.at<double>(0, i) = corners[i].x;
			vectors.at<double>(1, i) = corners[i].y;
		}
		cv::Mat res = H * vectors;
		for (size_t i = 0; i < 4; i++)
		{
			double h = res.at<double>(2, i);
			corners[i].x = res.at<double>(0, i) / h;
			corners[i].y = res.at<double>(1, i) / h;
		}
		cv::polylines(localFrame, corners, true, cv::Scalar(255, 255, 0));
		cv::imshow("frame", localFrame);
		cv::swap(prevFrame, grayFrame);
		prevKpts = filteredKpts;
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
	cv::Size subPixWinSize = { 10, 10 };
	cv::TermCriteria termcrit = { cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03 };
	corners = {
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
		int max_count = 500;
		double qualityLevel = 0.01;
		double minDistance = 10;
		cv::Mat mask = cv::Mat(height, width, CV_8U, cv::Scalar(0));
		cv::rectangle(mask, cv::Rect(x, y, w, h), cv::Scalar(255), -1);
		cv::goodFeaturesToTrack(localFrame, prevKpts, max_count, qualityLevel, minDistance, mask, 3, 3, 0, 0.04);
		cv::cornerSubPix(localFrame, prevKpts, subPixWinSize, cv::Size(-1, -1), termcrit);
		prevFrame = localFrame;
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
