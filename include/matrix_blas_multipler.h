#ifndef MATRIX_BLAS_MULTIPLIER_H
#define MATRIX_BLAS_MULTIPLIER_H

extern "C" {
#include <cblas.h>
}

#include <klepsydra/matrix_mult_benchmark/matrix_multiplier.h>

namespace kpsr {
namespace matrix_mult_benchmark {

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixBlasMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;

        multiplyHelper(A.data[0], B.data[0], C.data[0]);
    }

    virtual ~MatrixBlasMultiplier() {}

private:
    void multiplyHelper(const double *A_data, const double *B_data, double *C_data)
    {
        double alpha = 1.0f;
        double beta = 0.0f;
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    colsLeft,
                    rowsLeft,
                    rowsRight,
                    alpha,
                    A_data,
                    colsLeft,
                    B_data,
                    rowsLeft,
                    beta,
                    C_data,
                    rowsRight);
    }

    void multiplyHelper(const float *A_data, const float *B_data, float *C_data)
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    colsLeft,
                    rowsRight,
                    rowsLeft,
                    alpha,
                    A_data,
                    colsLeft,
                    B_data,
                    rowsRight,
                    beta,
                    C_data,
                    rowsRight);
    }

    void multiplyHelper(const int *A_data, const int *B_data, int *C_data)
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    colsLeft,
                    rowsLeft,
                    rowsRight,
                    alpha,
                    (float *) A_data,
                    colsLeft,
                    (float *) B_data,
                    rowsLeft,
                    beta,
                    (float *) C_data,
                    rowsRight);
    }
};
} // namespace matrix_mult_benchmark
} // namespace kpsr

#endif
