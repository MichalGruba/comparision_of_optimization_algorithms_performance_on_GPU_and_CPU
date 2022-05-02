#pragma once
struct Point;
double calc_fun(double x, double y, int fun);
Point initP(double maxx, double minx);
Point updateP(Point curr, Point best, double rk);
void spiralCPU(int fun, int search_points, int max_iter, int save);
void cvtToarray(Point* points, double* outX, double* outY, double* outZ, int fun, int size);
