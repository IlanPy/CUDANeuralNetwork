#pragma once

typedef struct {
	int rows;
	int cols;
	double** entries;
	double *device_entries;
	// bool is_device;
} Matrix;

Matrix* matrix_create(int row, int col);
Matrix* matrix_create_device(int row, int col); // Added
void matrix_copy_to_device(Matrix* host, Matrix* device); // Added
void matrix_copy_to_host(Matrix* device, Matrix* host); // Added
void matrix_free_device(Matrix* m); // Added
void matrix_fill(Matrix *m, int n);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m);
Matrix* matrix_copy(Matrix *m);
void matrix_save(Matrix* m, char* file_string);
Matrix* matrix_load(char* file_string);
void matrix_randomize(Matrix* m, int n);
int matrix_argmax(Matrix* m);
Matrix* matrix_flatten(Matrix* m, int axis);
