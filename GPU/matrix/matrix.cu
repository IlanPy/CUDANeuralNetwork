#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 100

Matrix* matrix_create(int row, int col) {
	//Matrix *matrix = malloc(sizeof(Matrix));
	Matrix* matrix = (Matrix*) malloc(sizeof(Matrix));
	matrix->rows = row;
	matrix->cols = col;
	// matrix->entries = malloc(row * sizeof(double*));
	matrix->entries = (double*) malloc(row * col * sizeof(double));
	return matrix;
}

void matrix_fill(Matrix *m, int n) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			// m->entries[i][j] = n;
			m->entries[i * m->cols + j] = n;
		}
	}
}

void matrix_free(Matrix *m) {
	free(m->entries);
	free(m);
	m = NULL;
}

void matrix_print(Matrix* m) {
	printf("Rows: %d Columns: %d\n", m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%1.3f ", m->entries[i * m->cols + j]);
		}
		printf("\n");
	}
}

Matrix* matrix_copy(Matrix* m) {
	Matrix* mat = matrix_create(m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			// mat->entries[i][j] = m->entries[i][j];
			mat->entries[i * mat->cols + j] = m->entries[i * m->cols + j];
		}
	}
	return mat;
}

void matrix_save(Matrix* m, char* file_string) {
	FILE* file = fopen(file_string, "w");
	fprintf(file, "%d\n", m->rows);
	fprintf(file, "%d\n", m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			fprintf(file, "%.6f\n", m->entries[i * m->cols + j]);
		}
	}
	printf("Successfully saved matrix to %s\n", file_string);
	fclose(file);
}

Matrix* matrix_load(char* file_string) {
	FILE* file = fopen(file_string, "r");
	char entry[MAXCHAR]; 
	fgets(entry, MAXCHAR, file);
	int rows = atoi(entry);
	fgets(entry, MAXCHAR, file);
	int cols = atoi(entry);
	Matrix* m = matrix_create(rows, cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			fgets(entry, MAXCHAR, file);
			m->entries[i * m->cols + j] = strtod(entry, NULL);
		}
	}
	printf("Sucessfully loaded matrix from %s\n", file_string);
	fclose(file);
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
	double max = 1.0 / sqrt(n);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->entries[i * m->cols + j] = uniform_distribution(min, max);
		}
	}
}
/*
int matrix_argmax(Matrix* m) {
	// Expects a Mx1 matrix
	double max_score = 0;
	int max_idx = 0;
	for (int i = 0; i < m->rows; i++) {
    		double val = m->entries[i * m->cols + 0];
    		if (val > max_score) {
        		max_score = val;
        		max_idx = i;
    		}
	}
	return max_idx;
}
*/
int matrix_argmax(Matrix* m) {
    double max_score = m->entries[0];
    int max_idx = 0;
    for (int i = 1; i < m->rows * m->cols; i++) {
        if (m->entries[i] > max_score) {
            max_score = m->entries[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/*
Matrix* matrix_flatten(Matrix* m, int axis) {
	// Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
	Matrix* mat;
	if (axis == 0) {
		mat = matrix_create(m->rows * m->cols, 1);
	} else if (axis == 1) {
		mat = matrix_create(1, m->rows * m->cols);
	} else {
		printf("Argument to matrix_flatten must be 0 or 1");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			//if (axis == 0) mat->entries[i * m->cols + j][0] = m->entries[i][j];
			if (axis == 0) mat->entries[i * m->cols + j] = m->entries[i * m->cols + j];
			// else if (axis == 1) mat->entries[0][i * m->cols + j] = m->entries[i][j];
			else if (axis == 0) mat->entries[0 * m->cols + i * m->cols + j] = m->entries[i * m->cols + j];
		}
	}
	return mat;
}
*/
/*
Matrix* matrix_flatten(Matrix* m, int axis) {
    // Axis = 0 -> Column Vector (n x 1)
    // Axis = 1 -> Row Vector (1 x n)
    int total = m->rows * m->cols;
    Matrix* mat;

    if (axis == 0) {
        mat = matrix_create(total, 1);
    } else if (axis == 1) {
        mat = matrix_create(1, total);
    } else {
        printf("Argument to matrix_flatten must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < total; i++) {
        mat->entries[i] = m->entries[i]; // Copy entry-by-entry
    }

    return mat;
}
*/
Matrix* matrix_flatten(Matrix* m, int axis) {
    // Axis = 0 -> Column Vector (n x 1)
    // Axis = 1 -> Row Vector (1 x n)
    int total = m->rows * m->cols;
    Matrix* mat;

    if (axis == 0) {
        mat = matrix_create(total, 1);
    } else if (axis == 1) {
        mat = matrix_create(1, total);
    } else {
        printf("Argument to matrix_flatten must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            int flat_index = i * m->cols + j;
            mat->entries[flat_index] = m->entries[i * m->cols + j]; // Or access through get(m, i, j)
        }
    }

    return mat;
}

