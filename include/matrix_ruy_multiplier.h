#ifndef MATRIX_RUY_MULTIPLIER_H
#define MATRIX_RUY_MULTIPLIER_H

#include <klepsydra/matrix_mult_benchmark/matrix_multiplier.h>
#include <ruy/ruy.h>

namespace kpsr {
namespace matrix_mult_benchmark {

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixRuyMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        ruy::Matrix<T> A_r;
        ruy::MakeSimpleLayout(colsLeft, rowsLeft, ruy::Order::kRowMajor, A_r.mutable_layout());
        A_r.set_data(A.data[0]);

        ruy::Matrix<T> B_r;
        ruy::MakeSimpleLayout(rowsLeft, rowsRight, ruy::Order::kRowMajor, B_r.mutable_layout());
        B_r.set_data(B.data[0]);

        ruy::Matrix<T> C_r;
        ruy::MakeSimpleLayout(colsLeft, rowsRight, ruy::Order::kRowMajor, C_r.mutable_layout());
        C_r.set_data(C.data[0]);

        ruy::MulParams<T, T> mul_params;
        ruy::Mul(A_r, B_r, mul_params, &context, &C_r);
    }

private:
    ruy::Context context;
};

} // namespace matrix_mult_benchmark
} // namespace kpsr
#endif
