#ifndef MATRIX_EIGEN_MULTIPLIER_H
#define MATRIX_EIGEN_MULTIPLIER_H

#include <matrix_multiplier.h>

#include <eigen3/Eigen/Dense>

#include <atomic>

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixTemplatedEigenMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
    typedef Eigen::Matrix<T, colsLeft, rowsRight, Eigen::RowMajor> RowMatrixXT;

public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        Eigen::Map<const Eigen::Matrix<T, colsLeft, rowsLeft, Eigen::StorageOptions::ColMajor>>
            A_e(A.data[0], colsLeft, rowsLeft);
        Eigen::Map<const Eigen::Matrix<T, rowsLeft, rowsRight, Eigen::StorageOptions::RowMajor>>
            B_e(B.data[0], rowsLeft, rowsRight);

        Eigen::Map<RowMatrixXT>(&C.data[0][0], colsLeft, rowsRight) = A_e * B_e;
        numberOfMultiplications++;
    }

    virtual ~MatrixTemplatedEigenMultiplier() {}

    static std::atomic<int> numberOfMultiplications;
};

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class MatrixDynamicEigenMultiplier : public MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight>
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenRowMatrix;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenColMatrix;

public:
    void multiply(const Matrix<T, colsLeft, rowsLeft> &A,
                  const Matrix<T, rowsLeft, rowsRight> &B,
                  Matrix<T, colsLeft, rowsRight> &C) override
    {
        C.sequence = A.sequence;
        C.timestamp = A.timestamp;
        Eigen::Map<const EigenRowMatrix> A_e(A.data[0], colsLeft, rowsLeft);
        Eigen::Map<const EigenRowMatrix> B_e(B.data[0], rowsLeft, rowsRight);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C_e;
        C_e.resize(colsLeft, rowsRight);
        C_e = A_e * B_e;

        numberOfMultiplications++;
    }

    virtual ~MatrixDynamicEigenMultiplier() {}

    static std::atomic<int> numberOfMultiplications;
};

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
std::atomic<int> 
    MatrixTemplatedEigenMultiplier<T, colsLeft, rowsLeft, rowsRight>::numberOfMultiplications(0);

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
std::atomic<int> 
    MatrixDynamicEigenMultiplier<T, colsLeft, rowsLeft, rowsRight>::numberOfMultiplications(0);

#endif
