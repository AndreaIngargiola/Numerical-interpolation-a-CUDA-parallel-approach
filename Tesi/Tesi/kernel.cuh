#include "hpc.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t l_compute_constants(int n, double* x, double* y);
double* l_compute_codomain(int n, double* x, double* y, double* xs, double* ys, int res_size);
double* l_compute_lagrange(int n, double* x, double* y, double* xs, double* ys, int res_size);