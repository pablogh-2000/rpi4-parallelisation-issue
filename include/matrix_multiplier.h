#ifndef MATRIX_MULTIPLIER_H
#define MATRIX_MULTIPLIER_H

#include "matrix.h"


template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixMultiplier
{
public:
    virtual void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                          const Matrix<T, rowsLeft, rowsRight> &B,
                          Matrix<T, colsLeft, rowsRight> &C) = 0;

    virtual ~MatrixMultiplier() {}
};

#endif // MATRIX_MULTIPLIER_H
