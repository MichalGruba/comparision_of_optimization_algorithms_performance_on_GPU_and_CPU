#pragma once
#include <fstream>
void saveDATA(const std::string& filename, const double* data, size_t size, int iter);
void saveDATA(const std::string& filename, const float* data, size_t size, int iter);
