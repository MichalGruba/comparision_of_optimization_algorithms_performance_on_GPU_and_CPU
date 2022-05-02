#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
struct Point;
__device__ double calc_fun_GPU(double x, double y, int fun);
__global__ void initPkernel(Point* points, int minx, int maxx, int* bestidx, unsigned int tseed, int fun);
__global__ void updatePkernel(Point* points, int minx, int maxx, int* bestidx, int fun);
void spiralGPU(int fun, int search_points, int max_iter);
double final_calc_fun(double x, double y, int fun);