#ifndef MATRIX_OPERATION_EVENT_H
#define MATRIX_OPERATION_EVENT_H

#include <klepsydra/matrix_mult_benchmark/matrix.h>
#include <memory>

namespace kpsr {
namespace matrix_mult_benchmark {

template<class T, size_t cols, size_t rows>
class MatrixOperationEvent
{
public:
    MatrixOperationEvent(std::shared_ptr<Matrix<T, cols, 1>> left,
                         std::shared_ptr<Matrix<T, 1, rows>> right,
                         int repetitions)
        : left(left)
        , right(right)
        , repetitions(repetitions)
    {}

    std::shared_ptr<Matrix<T, cols, 1>> left;
    std::shared_ptr<Matrix<T, 1, rows>> right;
    int repetitions;
};
} // namespace matrix_mult_benchmark
} // namespace kpsr
#endif // MATRIX_OPERATION_EVENT_H
