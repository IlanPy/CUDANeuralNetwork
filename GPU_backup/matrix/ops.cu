#include "ops.h"
#include <stdlib.h>
#include <stdio.h>

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

Matrix* multiply(Matrix *m1, Matrix *m2) {
        if (check_dimensions(m1, m2)) {
                Matrix *m = matrix_create(m1->rows, m1->cols);
                for (int i = 0; i < m1->rows; i++) {
                        for (int j = 0; j < m2->cols; j++) {
                                m->entriesf[i * m->cols + j] = m1->entriesf[i * m1->cols + j]
                                        * m2->entriesf[i * m2->cols + j];
                        }
                }
                return m;
        } else {
                printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
                exit(1);
        }
}
 
Matrix* add(Matrix *m1, Matrix *m2) {
        if (check_dimensions(m1, m2)) {
                Matrix *m = matrix_create(m1->rows, m1->cols);
                for (int i = 0; i < m1->rows; i++) {
                        for (int j = 0; j < m2->cols; j++) {
                                m->entriesf[i * m->cols + j] = m1->entriesf[i * m1->cols + j]
                                        + m2->entriesf[i * m2->cols + j];
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
                                m->entriesf[i * m->cols + j] = m1->entriesf[i * m1->cols + j]
                                        - m2->entriesf[i * m2->cols + j];
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
                        mat->entriesf[i * mat->cols + j] = (*func)(m->entriesf[i * m->cols + j]);
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
                                        sum += m1->entriesf[i * m1->cols + k]
                                                * m2->entriesf[k * m2->cols + j];
                                }
                                m->entriesf[i * m->cols + j] = sum;
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
                        mat->entriesf[i * m->cols + j] *= n;
                }
        }
        return mat;
}

Matrix* addScalar(double n, Matrix* m) {
        Matrix* mat = matrix_copy(m);
        for (int i = 0; i < m->rows; i++) {
                for (int j = 0; j < m->cols; j++) {
                        mat->entriesf[i * m->cols + j] += n;
                }
        }
        return mat;
}

Matrix* transpose(Matrix* m) {
        Matrix* mat = matrix_create(m->cols, m->rows);
        for (int i = 0; i < m->rows; i++) {
                for (int j = 0; j < m->cols; j++) {
                        mat->entriesf[j * mat->cols + i] =
                                m->entriesf[i * m->cols + j];
                }
        }
        return mat;
}