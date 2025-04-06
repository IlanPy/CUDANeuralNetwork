#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

__global__ void multiply_kernel(double* a, double* b, double* out, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		out[idx] = a[idx] * b[idx];
	}
}

Matrix* multiply_gpu(Matrix* a_dev, Matrix* b_dev) {
	if (check_dimensions(m1, m2)) {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
	Matrix* out = matrix_create_device(a_dev->rows, a_dev->cols);
	int total = a_dev->rows * a_dev->cols;
	int threads = 256;
	int blocks = (total + threads - 1) / threads;

	multiply_kernel<<<blocks, threads>>>(a_dev->data, b_dev->data, out->data, total);
	cudaDeviceSynchronize();
	return out;
}

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