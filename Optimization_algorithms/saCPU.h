#pragma once
struct Point;
Point initPoint(int minx, int maxx, double Tmax);
Point newPoint(Point point, int minx, int maxx);
double fitnessFun(Point point);
bool swapcrit(double E, double Eprim, double c, double T);
void saCPU();