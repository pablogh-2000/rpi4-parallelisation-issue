#ifndef MATRIX_GEMM_EIGEN_MULTIPLIER_H
#define MATRIX_GEMM_EIGEN_MULTIPLIER_H

#include <klepsydra/matrix_mult_benchmark/matrix_multiplier.h>

#include <Eigen/Dense>

#include <atomic>

namespace kpsr {
namespace matrix_mult_benchmark {

template <class T, size_t rows>
class MatrixGemmEigenMultiplier : public MatrixMultiplier<T, rows> {
    typedef Eigen::Matrix<T, rows, rows, Eigen::RowMajor> RowMatrixXT;
    
public:
    void multiply(const Matrix<T, rows> & A, const Matrix<T, rows> & B, Matrix<T, rows> & C) override {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        Eigen::Map<RowMatrixXT> A_e(&A.data[0][0], rows, rows);

        Eigen::Map<RowMatrixXT> C_e(&C.data[0][0], rows, rows);
        C_e.noalias() += A_e * A_e;

        numberOfMultiplications++;
    }

    static std::atomic<int> numberOfMultiplications;
};
}
}

template <class T, size_t rows>
std::atomic<int> kpsr::matrix_mult_benchmark::MatrixGemmEigenMultiplier<T, rows>::numberOfMultiplications(0);

#endif
