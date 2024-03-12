#ifndef MATRIX_NEON_MULTIPLIER_H
#define MATRIX_NEON_MULTIPLIER_H

#include <arm_neon.h>
#include <klepsydra/matrix_mult_benchmark/matrix_multiplier.h>

namespace kpsr {
namespace matrix_mult_benchmark {

template<class T, size_t colsLeft, size_t rowsRight>
class MatrixNeonMultiplier : public MatrixMultiplier<T, colsLeft, 1, rowsRight>
{
public:
    MatrixNeonMultiplier()
        : _colsLeft_div_4(colsLeft / 4)
        , _colsLeft_mod_4(colsLeft % 4)
        , _rowsRight_div_4(rowsRight / 4)
        , _rowsRight_mod_4(rowsRight % 4)
    {}

    void multiply(const Matrix<T, colsLeft, 1> &A,
                  const Matrix<T, 1, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        unsigned long int j, k;
        float32x4_t vbs[_rowsRight_div_4];
        for (k = 0; k < _rowsRight_div_4; k++) {
            vbs[k] = vld1q_f32(&B.data[0][k * 4]);
        }

        float32x4_t vas[_colsLeft_div_4];
        for (k = 0; k < _colsLeft_div_4; k++) {
            vas[k] = vld1q_f32(&A.data[k * 4][0]);
        }

        for (j = 0; j < colsLeft; j++) {
            // Calculate the result of the 4*4 matrix block directly through neon
            float32x4_t va = vdupq_n_f32(A.data[j][0]);

            for (k = 0; k < _rowsRight_div_4; k++) {
                float32x4_t vc = vmulq_f32(va, vbs[k]);
                vst1q_f32(&C.data[j][k * 4], vc);
            }
        }
        if (_rowsRight_mod_4 == 1) {
            float temp[4];

            float32x4_t vb = vdupq_n_f32(B.data[0][rowsRight - 1]);
            for (k = 0; k < _colsLeft_div_4; k++) {
                float32x4_t vc = vmulq_f32(vas[k], vb);
                vst1q_f32(temp, vc);
                C.data[k * 4][rowsRight - 1] = temp[0];
                C.data[(k * 4) + 1][rowsRight - 1] = temp[1];
                C.data[(k * 4) + 2][rowsRight - 1] = temp[2];
                C.data[(k * 4) + 3][rowsRight - 1] = temp[3];
            }
        }
        if (_colsLeft_div_4 == 1) {
            C.data[colsLeft - 1][rowsRight - 1] = A.data[colsLeft - 1][0] *
                                                  B.data[0][rowsRight - 1];
        }

        numberOfMultiplications++;
    }

    virtual ~MatrixNeonMultiplier() {}

    static std::atomic<int> numberOfMultiplications;

private:
    unsigned long int _colsLeft_div_4;
    unsigned long int _colsLeft_mod_4;
    unsigned long int _rowsRight_div_4;
    unsigned long int _rowsRight_mod_4;
};
} // namespace matrix_mult_benchmark
} // namespace kpsr

template<class T, size_t colsLeft, size_t rowsRight>
std::atomic<int> kpsr::matrix_mult_benchmark::MatrixNeonMultiplier<T, colsLeft, rowsRight>::
    numberOfMultiplications(0);

#endif // MATRIX_NEON_MULTIPLIER_H
