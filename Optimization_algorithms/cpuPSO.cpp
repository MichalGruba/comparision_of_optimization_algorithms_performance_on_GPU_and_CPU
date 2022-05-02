#define _USE_MATH_DEFINES
#include "cpuPSO.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <string>
#include "saveDATA.h"

const int dim = 2;
struct Particle {
	float position[dim];
	float velocity[dim];
	float fitness;
	float best_pos[dim];
	float best_fitness;
};
double fit(float position[], int fun) {
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
Particle init(float minx, float maxx, int fun) {
	Particle swarm;
	std::random_device rd;
	std::default_random_engine generator{ rd() };
	std::uniform_real_distribution<> rozklad(minx, maxx);
	for (int j = 0; j < dim; j++) {
		swarm.position[j] = rozklad(generator);
		swarm.best_pos[j] = swarm.position[j];
	}
	for (int j = 0; j < dim; j++) {
		swarm.velocity[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 100));;
	}
	swarm.fitness = fit(swarm.position, fun);
	swarm.best_fitness = swarm.fitness;
	return swarm;
	//-107374176.
}
void cvtToarray(Particle* points, float* outX, float* outY, float* outZ, int fun, int size) {
	for (int i = 0; i < size; i++) {
		outX[i] = points[i].position[0];
		outY[i] = points[i].position[1];
		outZ[i] = fit(points[i].position, fun);
	}
}
void startCPU(int fun, int NUM_OF_PARTICLES, int max_iter, int save) {
	srand(time(NULL));
	float w = 0.5;
	float c1 = 1.2;
	float c2 = 1.2;
	int iter = 0;

	float maxx = 100;
	float minx = -100;
	float maxv = 100;
	float minv = -100;
	Particle *swarm = new Particle[NUM_OF_PARTICLES];
	float best_global_pos[dim];
	float best_swarm_fitnessVal = INFINITY;

	float* psoCPU_X, * psoCPU_Y, * psoCPU_Z;
	int sizeCORD = NUM_OF_PARTICLES * sizeof(float);
	psoCPU_X = (float*)malloc(sizeCORD);
	psoCPU_Y = (float*)malloc(sizeCORD);
	psoCPU_Z = (float*)malloc(sizeCORD);

	for (int i = 0; i < NUM_OF_PARTICLES; i++) {
		if (i == NUM_OF_PARTICLES - 1) {
			int z = 1;
		}
		swarm[i] = init(minx,maxx,fun);
		if (swarm[i].fitness < best_swarm_fitnessVal) {
			best_swarm_fitnessVal = swarm[i].fitness;
			for (int k = 0; k < dim; k++) best_global_pos[k] = swarm[i].position[k];
		}
	}
	while (iter < max_iter) {
		for (int i = 0; i < NUM_OF_PARTICLES; i++) {

			for (int k = 0; k < dim; k++) {
				float r1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				float r2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				swarm[i].velocity[k] = (w * swarm[i].velocity[k] + c1 * r1 * (swarm[i].best_pos[k] - swarm[i].position[k])) + (c2 * r2 * (best_global_pos[k] - swarm[i].position[k]));
				if (swarm[i].velocity[k] < minv) swarm[i].velocity[k] = minv;
				else if (swarm[i].velocity[k] > maxv) swarm[i].velocity[k] = maxv;

				swarm[i].position[k] += swarm[i].velocity[k];
			}
			swarm[i].fitness = fit(swarm[i].position, fun);

			if (swarm[i].fitness < swarm[i].best_fitness) {
				swarm[i].best_fitness = swarm[i].fitness;
				for (int k = 0; k < dim; k++) swarm[i].best_pos[k] = swarm[i].position[k];
			}

			if (swarm[i].fitness < best_swarm_fitnessVal) {
				best_swarm_fitnessVal = swarm[i].fitness;
				for (int k = 0; k < dim; k++) best_global_pos[k] = swarm[i].position[k];
			}
		}
		if (save == 1) {
			cvtToarray(swarm, psoCPU_X, psoCPU_Y, psoCPU_Z, fun, NUM_OF_PARTICLES);
			std::string s = std::to_string(fun);
			saveDATA("psoCPU_X_fun_" + s + ".txt", psoCPU_X, NUM_OF_PARTICLES, iter);
			saveDATA("psoCPU_Y_fun_" + s + ".txt", psoCPU_Y, NUM_OF_PARTICLES, iter);
			saveDATA("psoCPU_Z_fun_" + s + ".txt", psoCPU_Z, NUM_OF_PARTICLES, iter);
		}
		iter += 1;
	}
	printf("x = %.4f, y = %.4f ", best_global_pos[0], best_global_pos[1]);
	printf("z = %.4f \n", best_swarm_fitnessVal);
}