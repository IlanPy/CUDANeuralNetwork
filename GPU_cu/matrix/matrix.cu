#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 100

__global__ void kernel_fill(double* data, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

__global__ void kernel_copy(const double* src, double* dst, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		dst[idx] = src[idx];
	}
}

__global__ void kernel_flatten(const double* src, double* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

Matrix* matrix_create(int row, int col) {
	Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
	matrix->rows = row;
	matrix->cols = col;

	matrix->entries = (double**) malloc(row * sizeof(double*));
	for (int i = 0; i < row; i++) {
	 	matrix->entries[i] = (double*) malloc(col * sizeof(double));
	}

	// Allocate entriesf on GPU
	cudaMalloc(&(matrix->entriesf), row * col * sizeof(double));
	
	return matrix;
}

void matrix_fill(Matrix *m, int n) {
    int size = m->rows * m->cols;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel_fill<<<blocks, threads>>>(m->entriesf, n, size);
    cudaDeviceSynchronize();
}

void matrix_free(Matrix *m) {
	for (int i = 0; i < m->rows; i++) {
		free(m->entries[i]);
	}
	free(m->entries);

	// Free GPU memory
	cudaFree(m->entriesf);

	free(m);
}

void matrix_print(Matrix* m) {
    int size = m->rows * m->cols;
    double* temp = (double*) malloc(size * sizeof(double));
    cudaMemcpy(temp, m->entriesf, size * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%1.3f ", temp[i * m->cols + j]);
        }
        printf("\n");
    }
    free(temp);
}

Matrix* matrix_copy(Matrix* m) {
	Matrix* mat = matrix_create(m->rows, m->cols);
	int size = m->rows * m->cols;
	int threads = 256;
	int blocks = (size + threads - 1) / threads;
	kernel_copy<<<blocks, threads>>>(m->entriesf, mat->entriesf, size);
	cudaDeviceSynchronize();
	return mat;
}

void matrix_save(Matrix* m, char* file_string) {
	FILE* file = fopen(file_string, "w");
	fprintf(file, "%d\n", m->rows);
	fprintf(file, "%d\n", m->cols);

	int size = m->rows * m->cols;
	double* host_data = (double*) malloc(size * sizeof(double));
	matrix_copy_device_to_host(m, host_data);

	for (int i = 0; i < size; i++) {
		fprintf(file, "%.6f\n", host_data[i]);
	}
	free(host_data);
	fclose(file);
	printf("Successfully saved matrix to %s\n", file_string);
}

Matrix* matrix_load(char* file_string) {
	FILE* file = fopen(file_string, "r");
	char entry[MAXCHAR];
	fgets(entry, MAXCHAR, file);
	int rows = atoi(entry);
	fgets(entry, MAXCHAR, file);
	int cols = atoi(entry);

	Matrix* m = matrix_create(rows, cols);
	int size = rows * cols;
	double* host_data = (double*) malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		fgets(entry, MAXCHAR, file);
		host_data[i] = strtod(entry, NULL);
	}
	fclose(file);
	matrix_copy_host_to_device(m, host_data);
	free(host_data);
	printf("Successfully loaded matrix from %s\n", file_string);
	return m;
}

double uniform_distribution(double low, double high) {
	double difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}

void matrix_randomize(Matrix* m, int n) {
    // Pulling from a random distribution of 
	// Min: -1 / sqrt(n)
	// Max: 1 / sqrt(n)
    double min = -1.0 / sqrt(n);
    double max =  1.0 / sqrt(n);
    int total = m->rows * m->cols;

    double* host_data = (double*) malloc(total * sizeof(double));
    for (int i = 0; i < total; i++) {
        host_data[i] = uniform_distribution(min, max);
    }

    matrix_copy_host_to_device(m, host_data);
    free(host_data);
}

int matrix_argmax(Matrix* m) {
	int size = m->rows * m->cols;
	double* host_copy = (double*) malloc(size * sizeof(double));
	matrix_copy_device_to_host(m, host_copy);

	double max_score = host_copy[0];
	int max_idx = 0;
	for (int i = 1; i < m->rows; i++) {
		if (host_copy[i] > max_score) {
			max_score = host_copy[i];
			max_idx = i;
		}
	}
	free(host_copy);
	return max_idx;
}

Matrix* matrix_flatten(Matrix* m, int axis) {
    Matrix* mat;
    int size = m->rows * m->cols;

    if (axis == 0) {
        mat = matrix_create(size, 1);
    } else if (axis == 1) {
        mat = matrix_create(1, size);
    } else {
        printf("Argument to matrix_flatten must be 0 or 1");
        exit(EXIT_FAILURE);
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel_flatten<<<blocks, threads>>>(m->entriesf, mat->entriesf, size);
    cudaDeviceSynchronize();

    return mat;
}

void matrix_copy_device_to_host(Matrix* m, double* host_data) {
    int size = m->rows * m->cols * sizeof(double);
    cudaMemcpy(host_data, m->entriesf, size, cudaMemcpyDeviceToHost);
}

void matrix_copy_host_to_device(Matrix* m, double* host_data) {
    int size = m->rows * m->cols * sizeof(double);
    cudaMemcpy(m->entriesf, host_data, size, cudaMemcpyHostToDevice);
}