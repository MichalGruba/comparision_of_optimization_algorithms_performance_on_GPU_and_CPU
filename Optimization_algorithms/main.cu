#define _USE_MATH_DEFINES
#include "cpuPSO.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "gpuPSO.cuh"
#include <chrono>
#include "spiralCPU.h"
#include "spiralGPU.cuh"
#include <iostream>
#include <fstream>
void savedata(const std::string& filename, const double* data, size_t size)
{
	std::ofstream out(filename);
	for (size_t i = 0; i < size; ++i)
	{
		out << data[i] << ((i + 1 < size) ? "," : "") << std::endl;
	}
	out.close();
}
int main() {
	cudaFree(0);
	std::cout << "Enter number of search points (int): ";
	int s_points;
	std::cin >> s_points;
	std::cout << "Enter number of iterations (int): ";
	int max_iter;
	std::cin >> max_iter;
	std::cout << "Choose function: " << std::endl;
	std::cout << "0: x^2 + y^2" << std::endl;
	std::cout << "1: (x + 2y - 7)^2 + (2x + y - 5)^2" << std::endl;
	std::cout << "2: 20 + x^2 - 10cos(2 * pi * x) + y^2 - 10cos(2 * pi * y)" << std::endl;
	std::cout << "3: (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2" << std::endl;
	std::cout << "4: [1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)] * [30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]" << std::endl;
	int fun;
	std::cin >> fun;
	std::cout << "Save points?" << std::endl;
	std::cout << "0: No" << std::endl;
	std::cout << "1: Yes" << std::endl;
	int save;
	std::cin >> save;
	const int N = 1;

	double SOACPU[N];
	double SOAGPU[N];
	double PSOCPU[N];
	double PSOGPU[N];
	int lock[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < N; i++) {
		printf("\n iter %i / %i \n", i, N);
		//////////////////////////////////////////////////
		/////////Spiral Optimization Algorithm GPU///////
		//////////////////////////////////////////////////
		auto t = std::chrono::high_resolution_clock::now();
		if (lock[0] == 0) spiralGPU(fun, s_points, max_iter);
		auto t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> timespent = t2 - t;
		if (timespent.count() > 120) lock[0] = 1;
		printf("(SOA GPU time: %f milliseconds).\n", timespent * 1000);
		SOAGPU[i] = timespent.count() * 1000;
		//////////////////////////////////////////////////
		/////////Spiral Optimization Algorithm CPU////////
		//////////////////////////////////////////////////
		t = std::chrono::high_resolution_clock::now();
		if(lock[1] == 0) spiralCPU(fun, s_points, max_iter, save);
		t2 = std::chrono::high_resolution_clock::now();
		timespent = t2 - t;
		if (timespent.count() > 120) lock[1] = 1;
		printf("(SOA CPU time: %f milliseconds).\n", timespent * 1000);
		SOACPU[i] = timespent.count() * 1000;
		//////////////////////////////////////////////////
		/////////Particle Swarm Optimization GPU//////////
		//////////////////////////////////////////////////
		t = std::chrono::high_resolution_clock::now();
		if (lock[2] == 0) startGPU(fun, s_points, max_iter);
		t2 = std::chrono::high_resolution_clock::now();
		timespent = t2 - t;
		if (timespent.count() > 120) lock[2] = 1;
		printf("(PSO GPU time: %f milliseconds).\n", timespent * 1000);
		PSOGPU[i] = timespent.count() * 1000;
		//////////////////////////////////////////////////
		/////////Particle Swarm Optimization CPU//////////
		//////////////////////////////////////////////////
		t = std::chrono::high_resolution_clock::now();
		if (lock[3] == 0) startCPU(fun, s_points, max_iter, save);
		t2 = std::chrono::high_resolution_clock::now();
		timespent = t2 - t;
		if (timespent.count() > 120) lock[3] = 1;
		printf("(PSO CPU time: %f milliseconds).\n", timespent * 1000);
		PSOCPU[i] = timespent.count()*1000;
	}
}