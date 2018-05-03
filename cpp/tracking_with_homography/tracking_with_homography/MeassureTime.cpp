#include "MeassureTime.h"



MeassureTime::MeassureTime() : start(std::chrono::high_resolution_clock::now())
{
}


MeassureTime::~MeassureTime()
{
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << (1e6 / duration) << std::endl;
}
