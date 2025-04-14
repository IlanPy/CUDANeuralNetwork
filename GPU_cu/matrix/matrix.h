#pragma once

typedef struct {
	double** entries;
	double* entriesf;
	int rows;
	int cols;
} Matrix;

Matrix* matrix_create(int row, int col);
void matrix_fill(Matrix *m, int n);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m);
Matrix* matrix_copy(Matrix *m);
void matrix_save(Matrix* m, char* file_string);
Matrix* matrix_load(char* file_string);
void matrix_randomize(Matrix* m, int n);
int matrix_argmax(Matrix* m);
Matrix* matrix_flatten(Matrix* m, int axis);
void matrix_copy_device_to_host(Matrix* m, double* host_data);
void matrix_copy_host_to_device(Matrix* m, double* host_data);
