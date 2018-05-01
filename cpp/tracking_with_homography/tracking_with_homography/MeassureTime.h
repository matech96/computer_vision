#pragma once
#include <chrono>
#include <iostream>
class MeassureTime
{
	std::chrono::high_resolution_clock::time_point start;
public:
	MeassureTime();
	~MeassureTime();
};

