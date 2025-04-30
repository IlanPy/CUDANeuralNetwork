#include "activations.h"

#include <math.h>
#include "../matrix/ops.h"

/*
__global__ void kernel_exp(double* input, double* output, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		output[idx] = exp(input[idx]);
	}
}
*/
__global__ void kernel_normalize(double* input, double* output, double total, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = exp(input[idx]) / total;
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

    Matrix* mat = matrix_create(m->rows, m->cols);

    double total = 0.0;
    for (int i = 0; i < size; i++) {
        total += exp(m->entriesf[i]);
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel_normalize<<<blocks, threads>>>(
        m->entriesf, mat->entriesf, total, size
    );
    cudaDeviceSynchronize();

    return mat;
}

