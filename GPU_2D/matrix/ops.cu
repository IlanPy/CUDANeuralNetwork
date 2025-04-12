#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

__global__ void elementwise_multiply_kernel(double* A, double* B, double* C, int total) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] * B[idx];
    }
}

Matrix* multiply(Matrix* m1, Matrix* m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mismatch\n");
        exit(1);
    }

    int total = m1->rows * m1->cols;
    size_t bytes = total * sizeof(double);

    double* h_A = &m1->entries[0][0];
    double* h_B = &m2->entries[0][0];

    Matrix* mat = matrix_create(m1->rows, m1->cols);
    // double* h_C = &mat->entries[0][0];

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    elementwise_multiply_kernel<<<blocks, threads>>>(d_A, d_B, d_C, total);

    // cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&mat->entries[0][0], d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return mat;
}


/*
__global__ void elementwise_multiply_kernel(double* A, double* B, double* C, int total) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] * B[idx];
    }
}

Matrix* multiply(Matrix* m1, Matrix* m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mismatch: %dx%d vs %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(EXIT_FAILURE);
    }

    int total = m1->rows * m1->cols;
    size_t bytes = total * sizeof(double);

    // Get flat view of data
    double* h_A = m1->entries[0];
    double* h_B = m2->entries[0];

    // Allocate result matrix (will also be structured like m1/m2)
    Matrix* result = matrix_create(m1->rows, m1->cols);
    double* h_C = result->entries[0];  // flat pointer to result buffer

    // Device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    elementwise_multiply_kernel<<<blocks, threads>>>(d_A, d_B, d_C, total);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}
*/

/*
Matrix* multiply(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}
*/
Matrix* add(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix *mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	return mat;
}

Matrix* dot(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] *= n;
		}
	}
	return mat;
}

Matrix* addScalar(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] += n;
		}
	}
	return mat;
}

Matrix* transpose(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = m->entries[i][j];
		}
	}
	return mat;
}
