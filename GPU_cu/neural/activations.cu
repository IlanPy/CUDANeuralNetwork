#include "activations.h"

#include <math.h>
#include "../matrix/ops.h"

__global__ void kernel_exp(double* input, double* output, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		output[idx] = exp(input[idx]);
	}
}

__global__ void kernel_normalize(double* data, double total, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx] /= total;
	}
}

double sigmoid(double input) {
	return 1.0 / (1 + exp(-1 * input));
}

Matrix* sigmoidPrime(Matrix* m) {
	Matrix* ones = matrix_create(m->rows, m->cols);
	matrix_fill(ones, 1);
	Matrix* subtracted = subtract(ones, m);
	Matrix* multiplied = multiply(m, subtracted);
	matrix_free(ones);
	matrix_free(subtracted);
	return multiplied;
}

Matrix* softmax(Matrix* m) {
	int size = m->rows * m->cols;
	
	double* exp_vals;
	cudaMalloc(&exp_vals, size * sizeof(double));

	int blockSize = 256;
	int gridSize = (size + blockSize - 1) / blockSize;
	kernel_exp<<<gridSize, blockSize>>>(m->entriesf, exp_vals, size);
	cudaDeviceSynchronize();
	
	double* host_exp_vals = (double*) malloc(size * sizeof(double));
	cudaMemcpy(host_exp_vals, exp_vals, size * sizeof(double), cudaMemcpyDeviceToHost);
	
	double total = 0.0;
	for (int i = 0; i < size; i++) total += host_exp_vals[i];

	kernel_normalize<<<gridSize, blockSize>>>(exp_vals, total, size);
	cudaDeviceSynchronize();

	Matrix* mat = matrix_create(m->rows, m->cols);
	cudaMemcpy(mat->entriesf, exp_vals, size * sizeof(double), cudaMemcpyDeviceToDevice);

	free(host_exp_vals);
	cudaFree(exp_vals);

	return mat;
}

