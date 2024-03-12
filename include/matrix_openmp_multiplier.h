#ifndef MATRIX_OPENMP_MULTIPLIER_H
#define MATRIX_OPENMP_MULTIPLIER_H

#include <omp.h>

#include <matrix_multiplier.h>

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixOmpMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        size_t i;
        size_t j;
        size_t k;
#pragma omp parallel shared(A, B, C) private(i, j, k)
        {
#pragma omp for
            for (i = 0; i < colsLeft; i++) {
                for (j = 0; j < rowsRight; j++) {
                    C.data[i][j] = 0.0;
                    for (k = 0; k < rowsLeft; k++) {
                        C.data[i][j] = C.data[i][j] + A.data[i][k] * B.data[k][j];
                    }
                }
            }
        }
    }

    virtual ~MatrixOmpMultiplier() {}
};

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixOmpSimdMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        size_t i;
        size_t j;
        size_t k;
#pragma omp parallel shared(A, B, C) private(i, j, k)
        {
#pragma omp for simd
            for (i = 0; i < colsLeft; i++) {
                for (j = 0; j < rowsRight; j++) {
                    C.data[i][j] = 0.0;
                    for (k = 0; k < rowsLeft; k++) {
                        C.data[i][j] = C.data[i][j] + A.data[i][k] * B.data[k][j];
                    }
                }
            }
        }
    }

    virtual ~MatrixOmpSimdMultiplier() {}
};

#endif // MATRIX_OPENMP_MULTIPLIER_H
