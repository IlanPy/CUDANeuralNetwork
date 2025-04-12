#pragma once

typedef struct {
	double** entries;
	double* entriesf;
	int rows;
	int cols;
} Matrix;

//void matrix_diff_self(Matrix* m, char* label);
Matrix* matrix_create(int row, int col);
void matrix_fill(Matrix *m, int n);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m);
Matrix* matrix_copy(Matrix *m);
// Matrix* matrix_copy_2d(Matrix* m);
void matrix_save(Matrix* m, char* file_string);
Matrix* matrix_load(char* file_string);
void matrix_randomize(Matrix* m, int n);
int matrix_argmax(Matrix* m);
Matrix* matrix_flatten(Matrix* m, int axis);
