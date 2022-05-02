#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
struct Particle;
__device__ double fitGPU(float position[], int fun);
__device__ Particle initGPU(curandState state, float minx, float maxx, float minv, float maxv, int fun);
__global__ void mainGPU(float* pos, float* fitness, Particle* swarm, int fun, int max_iter, int NUM_OF_PARTICLES);
void startGPU(int fun, int search_points, int max_iter);