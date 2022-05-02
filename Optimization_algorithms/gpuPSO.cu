
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

const int THREADS_PER_BLOCK = 512;
const int dim = 2;
struct Particle {
	float position[dim];
	float velocity[dim];
	float fitness;
	float best_pos[dim];
	float best_fitness;
};
__device__ double fitGPU(float position[], int fun) {
	float x = position[0];
	float y = position[1];

	double f;
	if (fun == 0) f = pow(x, 2) + pow(y, 2);
	else if (fun == 1) f = pow((x + 2 * y - 7), 2) + pow((2 * x + y - 5), 2);
	else if (fun == 2) f = 20 + pow(x, 2) - 10 * cos(2 * M_PI * x) + pow(y, 2) - 10 * cos(2 * M_PI * y);
	else if (fun == 3) f = pow(1.5 - x + x * y, 2) + pow(2.25 - x + x * pow(y, 2), 2) + pow(2.625 - x + x * pow(y, 3), 2);
	else if (fun == 4) f = (1 + pow(x + y + 1, 2) * (19 - 14 * x + 3 * pow(x, 2) - 14 * y + 6 * x * y + 3 * pow(y, 2))) * (30 + pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * pow(x, 2) + 48 * y - 36 * x * y + 27 * pow(y, 2)));
	else f = pow((x + 2 * y - 7), 2) + pow((2 * x + y - 5), 2);
	return f;
}
__device__ Particle initGPU(curandState state, float minx, float maxx, float minv, float maxv, int fun) {
	Particle particle;
	for (int j = 0; j < dim; j++) {
		particle.position[j] = curand_uniform(&state)*(maxx - minx) + minx;
		particle.best_pos[j] = particle.position[j];
	}
	for (int j = 0; j < dim; j++) {
		particle.velocity[j] = curand_uniform(&state)*(maxv - minv) + minv;
	}
	particle.fitness = fitGPU(particle.position, fun);
	particle.best_fitness = particle.fitness;
	return particle;
}
__device__ float global_best_pos[dim];
__device__ float global_best_fitness;
__global__ void mainGPU(float *pos, float *fitness, Particle *swarm, int fun, int max_iter, int NUM_OF_PARTICLES) {
	curandState state;
	//inertia
	float w = 0.5;
	//coefficients
	float c1 = 1.2;
	float c2 = 1.2;
	//iterations
	int iter = 0;
	//int max_iter = 1000;
	//search domain
	float maxx = 100;
	float minx = -100;
	//velocity limits
	float maxv = 100;
	float minv = -100;
	unsigned int seed = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, 0, 0, &state);
	global_best_fitness = *fitness;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//initiate particles
	swarm[i] = initGPU(state,minx,maxx,minv,maxv,fun);
	__syncthreads();
	if (swarm[i].fitness < global_best_fitness) {
		global_best_fitness = swarm[i].fitness;
		for (int k = 0; k < dim; k++) global_best_pos[k] = swarm[i].position[k];
	}

	while (iter < max_iter) {

		for (int k = 0; k < dim; k++) {
			float r1 = curand_uniform(&state);
			float r2 = curand_uniform(&state);
			swarm[i].velocity[k] = (w * swarm[i].velocity[k] + c1 * r1 * (swarm[i].best_pos[k] - swarm[i].position[k])) + (c2 * r2 * (global_best_pos[k] - swarm[i].position[k]));
			if (swarm[i].velocity[k] < minv) swarm[i].velocity[k] = minv;
			else if (swarm[i].velocity[k] > maxv) swarm[i].velocity[k] = maxv;

			swarm[i].position[k] += swarm[i].velocity[k];
		}
		swarm[i].fitness = fitGPU(swarm[i].position, fun);

		if (swarm[i].fitness < swarm[i].best_fitness) {
			swarm[i].best_fitness = swarm[i].fitness;
			for (int k = 0; k < dim; k++) swarm[i].best_pos[k] = swarm[i].position[k];
		}

		if (swarm[i].fitness < global_best_fitness) {
			global_best_fitness = swarm[i].fitness;
			for (int k = 0; k < dim; k++) global_best_pos[k] = swarm[i].position[k];
		}
		__syncthreads();
		iter += 1;
	}
	for(int k = 0; k<dim;k++) pos[k] = global_best_pos[k];
	*fitness = global_best_fitness;

}

void startGPU(int fun, int NUM_OF_PARTICLES, int max_iter) {
	srand(time(NULL));
	float *d_pos;
	float* fitness, * d_fitness;
	Particle* swarm;
	Particle* d_swarm;
	int sizeFLOAT = sizeof(float);
	int sizeTAB = sizeof(float) * dim;
	int sizePART = sizeof(Particle) * NUM_OF_PARTICLES;

	float *pos = (float*)malloc(sizeTAB);
	swarm = (Particle*)malloc(sizePART);
	fitness = (float*)malloc(sizeFLOAT);
	for (int j = 0; j < dim; j++) {
		pos[j] = 10;
	}
	cudaMalloc((void**)&d_pos, sizeTAB);
	cudaMalloc((void**)&d_fitness, sizeFLOAT);
	cudaMalloc((void**)&d_swarm, sizePART);
	*fitness = 900000;
	cudaMemcpy(d_fitness, fitness, sizeFLOAT, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, pos, sizeTAB, cudaMemcpyHostToDevice);
	cudaMemcpy(d_swarm, swarm, sizePART, cudaMemcpyHostToDevice);
	mainGPU << <NUM_OF_PARTICLES/ THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_pos,d_fitness, d_swarm, fun, max_iter, NUM_OF_PARTICLES);
	//cudaDeviceSynchronize();
	//cudaError_t error = cudaGetLastError();
	//printf("CUDA error: %s\n", cudaGetErrorString(error));
	cudaMemcpy(fitness, d_fitness, sizeFLOAT, cudaMemcpyDeviceToHost);
	cudaMemcpy(pos, d_pos, sizeTAB, cudaMemcpyDeviceToHost);

	printf("x = %.4f, y = %.4f ", pos[0], pos[1]);
	printf("z = %.4f \n", *fitness);
	cudaFree(d_pos); cudaFree(d_fitness); cudaFree(d_swarm);
	free(pos); free(fitness); free(swarm);
}
