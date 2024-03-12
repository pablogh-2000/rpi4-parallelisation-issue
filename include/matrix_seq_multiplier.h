#ifndef MATRIX_SEQ_MULTIPLIER_H
#define MATRIX_SEQ_MULTIPLIER_H

#include "matrix_multiplier.h"

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixSeqMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        for (size_t i = 0; i < colsLeft; ++i) {
            for (size_t k = 0; k < rowsLeft; ++k) {
                for (size_t j = 0; j < rowsRight; ++j) {
                    C.data[i][j] *= (k != 0);
                    C.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
    }

    virtual ~MatrixSeqMultiplier() {}
};

#endif // MATRIX_SEQ_MULTIPLIER_H
