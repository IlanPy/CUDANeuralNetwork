#pragma once

#include "matrix.h"

Matrix* multiply_gpu(Matrix* a_dev, Matrix* b_dev);
Matrix* add(Matrix* m1, Matrix* m2);
Matrix* subtract(Matrix* m1, Matrix* m2);
Matrix* dot(Matrix* m1, Matrix* m2);
Matrix* apply(double (*func)(double), Matrix* m);
Matrix* scale(double n, Matrix* m);
Matrix* addScalar(double n, Matrix* m);
Matrix* transpose(Matrix* m);
