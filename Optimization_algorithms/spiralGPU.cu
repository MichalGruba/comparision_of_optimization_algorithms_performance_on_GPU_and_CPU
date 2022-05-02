#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

struct Point {
	double x;
	double y;
};

__device__ double calc_fun_GPU(double x, double y, int fun) {
	double f;
	if (fun == 0) f = pow(x, 2) + pow(y, 2);
	else if (fun == 1) f = pow((x + 2 * y - 7), 2) + pow((2 * x + y - 5), 2);
	else if (fun == 2) f = 20 + pow(x, 2) - 10 * cos(2 * M_PI * x) + pow(y, 2) - 10 * cos(2 * M_PI * y);
	else if (fun == 3) f = pow(1.5 - x + x * y, 2) + pow(2.25 - x + x * pow(y, 2), 2) + pow(2.625 - x + x * pow(y, 3), 2);
	else if (fun == 4) f = (1 + pow(x + y + 1, 2) * (19 - 14 * x + 3 * pow(x, 2) - 14 * y + 6 * x * y + 3 * pow(y, 2))) * (30 + pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * pow(x, 2) + 48 * y - 36 * x * y + 27 * pow(y, 2)));
	else {
		f = pow(x, 2) + pow(y, 2);
	}
	return f;
}
double final_calc_fun(double x, double y, int fun) {
	double f;
	if (fun == 0) f = 0.1 * pow(x, 2) + x + pow(y, 3) + 4 * pow(y, 2);
	else if (fun == 1) f = pow((x + 2 * y - 7), 2) + pow((2 * x + y - 5), 2);
	else if (fun == 2) f = 20 + pow(x, 2) - 10 * cos(2 * M_PI * x) + pow(y, 2) - 10 * cos(2 * M_PI * y);
	else if (fun == 3) f = pow(1.5 - x + x * y, 2) + pow(2.25 - x + x * pow(y, 2), 2) + pow(2.625 - x + x * pow(y, 3), 2);
	else if (fun == 4) f = (1 + pow(x + y + 1, 2) * (19 - 14 * x + 3 * pow(x, 2) - 14 * y + 6 * x * y + 3 * pow(y, 2))) * (30 + pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * pow(x, 2) + 48 * y - 36 * x * y + 27 * pow(y, 2)));
	else f = pow(x, 2) + pow(y, 2);
	return f;
}
__device__ float fbest;
__device__ int ib;

__global__ void initPkernel(Point *points, int minx, int maxx, int* bestidx, unsigned int tseed, int fun) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	unsigned int seed = blockIdx.x * blockDim.x + threadIdx.x + tseed;
	curand_init(seed, 0, 0, &state);
	points[i].x = curand_uniform(&state)* (maxx - minx) + minx;
	points[i].y = curand_uniform(&state) * (maxx - minx) + minx;
	double f = calc_fun_GPU(points[i].x, points[i].y, fun);
	if (i == 0) {
		fbest = f;
	}
	__syncthreads();

	if (i != 0 && f < fbest) {
		atomicExch(&ib, i);
		atomicExch(&fbest, f);
	}
	__syncthreads();
	*bestidx = ib;
}
__global__ void updatePkernel(Point* points, int minx, int maxx, int *bestidx, int fun) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	double rk = 0.95;

	double alpha = M_PI / 8;
	points[i].x = rk * (cos(alpha) * points[i].x - sin(alpha) * points[i].y) - (rk * points[ib].x * cos(alpha) - rk * sin(alpha) * points[ib].y - points[ib].x);
	points[i].y = rk * (sin(alpha) * points[i].x + cos(alpha) * points[i].y) - (rk * points[ib].x * sin(alpha) + rk * cos(alpha) * points[ib].y - points[ib].y);

	float f = calc_fun_GPU(points[i].x, points[i].y, fun);
	if (f < fbest) {
		atomicExch(&ib, i);
		atomicExch(&fbest, f);
	}
	__syncthreads();
	*bestidx = ib;
}

void spiralGPU(int fun, int search_points, int max_iter) {
	const int THREADS_PER_BLOCK = 512;
	int maxx = 5;
	int minx = -5;
	double f;
	int *ib;
	int* d_ib;
	double fbest;
	double rk;
	double delta = 0.0001;

	size_t sizePOINTS = search_points * sizeof(Point);
	Point* points, * d_points;
	points = (Point*)malloc(sizePOINTS);
	ib = (int*)malloc(sizeof(int));
	*ib = 0;
	cudaMalloc((void**)&d_points, sizePOINTS);
	cudaMalloc((void**)&d_ib, sizeof(int));
	cudaMemcpy(d_points, points, sizePOINTS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ib, ib, sizeof(int), cudaMemcpyHostToDevice);
	initPkernel << <search_points / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_points, minx, maxx,d_ib,time(NULL),fun);
	cudaMemcpy(ib, d_ib, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < max_iter; i++) {
		updatePkernel << <search_points / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_points, minx, maxx,d_ib,fun);
	}
	cudaMemcpy(ib, d_ib, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(points, d_points, sizePOINTS, cudaMemcpyDeviceToHost);
	double result = final_calc_fun(points[*ib].x, points[*ib].y, fun);
	printf("x = %.4f y = %.4f z = %.4f\n", points[*ib].x, points[*ib].y, result);
	cudaFree(d_points); cudaFree(d_ib);
	free(points); free(ib);
}