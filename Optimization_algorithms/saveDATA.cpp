#include <fstream>
void saveDATA(const std::string& filename, const double* data, size_t size, int iter)
{
	std::ofstream out;
	if (iter == 0) out.open(filename);
	if(iter > 0) out.open(filename, std::ios_base::app);
	for (size_t i = 0; i < size; ++i)
	{
		out << data[i] << ((i + 1 < size) ? "," : "") << std::endl;
	}
	out.close();
}

void saveDATA(const std::string& filename, const float* data, size_t size, int iter)
{
	std::ofstream out;
	if (iter == 0) out.open(filename);
	if (iter > 0) out.open(filename, std::ios_base::app);
	for (size_t i = 0; i < size; ++i)
	{
		out << data[i] << ((i + 1 < size) ? "," : "") << std::endl;
	}
	out.close();
}