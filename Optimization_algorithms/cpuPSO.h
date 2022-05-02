#pragma once
void startCPU(int fun, int NUM_OF_PARTICLES, int max_iter, int save);
struct Particle;
double fit(float position[], int fun);
Particle init(float minx, float maxx, int fun);
