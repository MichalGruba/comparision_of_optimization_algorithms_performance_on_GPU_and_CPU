#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <random>
#include <string>
#include "saveDATA.h"

struct Point {
	double x;
	double y;
};
double calc_fun(double x, double y, int fun) {
	double f;
	if (fun == 0) f = pow(x,2)+pow(y,2);
	else if (fun == 1) f = pow((x + 2 * y - 7), 2) + pow((2 * x + y - 5), 2);
	else if (fun == 2) f = 20 + pow(x, 2) - 10 * cos(2 * M_PI * x) + pow(y, 2) - 10 * cos(2 * M_PI * y);
	else if (fun == 3) f = pow(1.5 - x + x * y, 2) + pow(2.25 - x + x * pow(y, 2), 2) + pow(2.625 - x + x * pow(y, 3), 2);
	else if (fun == 4) f = (1 + pow(x + y + 1, 2) * (19 - 14 * x + 3 * pow(x, 2) - 14 * y + 6 * x * y + 3 * pow(y, 2))) * (30 + pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * pow(x, 2) + 48 * y - 36 * x * y + 27 * pow(y, 2)));
	else f = pow(x,2)+pow(y,2);
	return f;
}
Point initP(double maxx, double minx) {
	std::random_device rd;
	std::default_random_engine generator{ rd() };
	std::uniform_real_distribution<> rozklad(minx, maxx);
	Point p;
	p.x = rozklad(generator);
	p.y = rozklad(generator);
	return p;
}
Point updateP(Point curr, Point best, double rk) {
	Point next;
	double alpha = M_PI / 8;
	next.x = rk * (cos(alpha) * curr.x - sin(alpha) * curr.y) - (rk * best.x * cos(alpha) - rk * sin(alpha) * best.y - best.x);
	next.y = rk * (sin(alpha) * curr.x + cos(alpha) * curr.y) - (rk * best.x * sin(alpha) + rk * cos(alpha) * best.y - best.y);
	return next;
}
void cvtToarray(Point *points, double *outX, double* outY, double* outZ, int fun, int size) {
	for (int i = 0; i < size; i++) {
		outX[i] = points[i].x;
		outY[i] = points[i].y;
		outZ[i] = calc_fun(points[i].x, points[i].y, fun);
	}
}
void spiralCPU(int fun, int search_points, int max_iter, int save) {
	int maxx = 5;
	int minx = -5;
	double f;
	int ib = 0;
	double fbest;
	double rk;
	double delta = 0.0001;
	Point *points = new Point[search_points];

	double* spiralCPU_X, * spiralCPU_Y, * spiralCPU_Z;
	int sizeCORD = search_points * sizeof(double);
	spiralCPU_X = (double*)malloc(sizeCORD);
	spiralCPU_Y = (double*)malloc(sizeCORD);
	spiralCPU_Z = (double*)malloc(sizeCORD);

	for (int i = 0; i < search_points; i++) {
		points[i] = initP(maxx, minx);
		f = calc_fun(points[i].x, points[i].y, fun);
		if (i == 0) fbest = f;
		if (i != 0 && f < fbest) {
			ib = i;
			fbest = f;
		}
	}
	for (int i = 0; i < max_iter; i++) {
		rk = 0.95;
		for (int j = 0; j < search_points; j++) {
			points[j] = updateP(points[j], points[ib], rk);
			f = calc_fun(points[j].x, points[j].y, fun);
			if (f < fbest) {
				ib = j;
				fbest = f;
			}
		}
		if (save == 1) {
			cvtToarray(points, spiralCPU_X, spiralCPU_Y, spiralCPU_Z, fun, search_points);
			std::string s = std::to_string(fun);
			saveDATA("spiralCPU_X_fun_" + s + ".txt", spiralCPU_X, search_points, i);
			saveDATA("spiralCPU_Y_fun_" + s + ".txt", spiralCPU_Y, search_points, i);
			saveDATA("spiralCPU_Z_fun_" + s + ".txt", spiralCPU_Z, search_points, i);
		}
	}
	printf("x = %.4f y = %.4f z = %.4f\n", points[ib].x, points[ib].y, fbest);
}