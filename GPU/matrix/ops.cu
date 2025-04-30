#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}


__global__ void elementwise_op(double* A, double* B, double* C, int total, int op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        switch (op) {
            case 0: C[idx] = A[idx] + B[idx]; break;
            case 1: C[idx] = A[idx] - B[idx]; break;
            case 2: C[idx] = A[idx] * B[idx]; break;
        }
    }
}

__global__ void apply_scalar_kernel(double* A, double* C, int total, double n, int op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        if (op == 0) C[idx] = A[idx] * n;
        else C[idx] = A[idx] + n;
    }
}

__global__ void transpose_kernel(double* A, double* B, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int i = idx / cols;
        int j = idx % cols;
        B[j * rows + i] = A[i * cols + j];
    }
}

__global__ void dot_kernel(double* A, double* B, double* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

#define TILE_WIDTH 16

__global__ void tiled_dot_kernel(double* A, double* B, double* C, int m, int n, int p) {
    __shared__ double tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    double sum = 0.0;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < m && t * TILE_WIDTH + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (col < p && t * TILE_WIDTH + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = sum;
}

Matrix* multiply(Matrix *m1, Matrix *m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mismatch multiply\n");
        exit(1);
    }
    int total = m1->rows * m1->cols;
    Matrix* result = matrix_create(m1->rows, m1->cols);
    int threads = 256, blocks = (total + threads - 1) / threads;
    elementwise_op<<<blocks, threads>>>(m1->entriesf, m2->entriesf, result->entriesf, total, 2);
    return result;
}

Matrix* add(Matrix *m1, Matrix *m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mismatch add\n");
        exit(1);
    }
    int total = m1->rows * m1->cols;
    Matrix* result = matrix_create(m1->rows, m1->cols);
    int threads = 256, blocks = (total + threads - 1) / threads;
    elementwise_op<<<blocks, threads>>>(m1->entriesf, m2->entriesf, result->entriesf, total, 0);
    return result;
}

Matrix* subtract(Matrix *m1, Matrix *m2) {
    if (!check_dimensions(m1, m2)) {
        printf("Dimension mismatch subtract\n");
        exit(1);
    }
    int total = m1->rows * m1->cols;
    Matrix* result = matrix_create(m1->rows, m1->cols);
    int threads = 256, blocks = (total + threads - 1) / threads;
    elementwise_op<<<blocks, threads>>>(m1->entriesf, m2->entriesf, result->entriesf, total, 1);
    return result;
}

Matrix* apply(double (*func)(double), Matrix* m) {
        Matrix *mat = matrix_copy(m);
        for (int i = 0; i < m->rows; i++) {
                for (int j = 0; j < m->cols; j++) {
                        mat->entriesf[i * mat->cols + j] = (*func)(m->entriesf[i * m->cols + j]);
                }
        }
        return mat;
}

Matrix* dot(Matrix *m1, Matrix *m2) {
    if (m1->cols != m2->rows) {
        printf("Dimension mismatch dot\n");
        exit(1);
    }
    Matrix* result = matrix_create(m1->rows, m2->cols);
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((m2->cols + TILE_WIDTH - 1)/TILE_WIDTH, (m1->rows + TILE_WIDTH - 1)/TILE_WIDTH);
    dot_kernel<<<blocks, threads>>>(m1->entriesf, m2->entriesf, result->entriesf, m1->rows, m1->cols, m2->cols);
    // tiled_dot_kernel<<<blocks, threads>>>(m1->entriesf, m2->entriesf, result->entriesf, m1->rows, m1->cols, m2->cols);
    return result;
}

Matrix* scale(double n, Matrix* m) {
    int total = m->rows * m->cols;
    Matrix* result = matrix_copy(m);
    int threads = 256, blocks = (total + threads - 1) / threads;
    apply_scalar_kernel<<<blocks, threads>>>(m->entriesf, result->entriesf, total, n, 0);
    return result;
}

Matrix* addScalar(double n, Matrix* m) {
    int total = m->rows * m->cols;
    Matrix* result = matrix_copy(m);
    int threads = 256, blocks = (total + threads - 1) / threads;
    apply_scalar_kernel<<<blocks, threads>>>(m->entriesf, result->entriesf, total, n, 1);
    return result;
}

Matrix* transpose(Matrix* m) {
    Matrix* result = matrix_create(m->cols, m->rows);
    int total = m->rows * m->cols;
    int threads = 256, blocks = (total + threads - 1) / threads;
    transpose_kernel<<<blocks, threads>>>(m->entriesf, result->entriesf, m->rows, m->cols);
    return result;
}
