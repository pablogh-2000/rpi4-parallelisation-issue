#ifndef MATRIX_DATA_FACTORY_H
#define MATRIX_DATA_FACTORY_H

#include <functional>
#include <klepsydra/matrix_mult_benchmark/matrix.h>
#include <memory>

namespace kpsr {
namespace matrix_mult_benchmark {

enum MatrixType { IDENTITY = 0, CONSTANT, RANDOM };

template<class T, size_t cols, size_t rows>
class MatrixDataFactory
{
public:
    MatrixDataFactory(MatrixType type, const T &value, std::function<T()> randomGenerator)
        : _sequence(0)
        , _type(type)
        , _matrix(new Matrix<T, cols, rows>(_sequence))
        , _randomGenerator(randomGenerator)
    {
        for (size_t i = 0; i < cols; i++) {
            for (size_t j = 0; j < rows; j++) {
                if (type == IDENTITY) {
                    if (i == j) {
                        _matrix->data[i][j] = 1;
                    } else {
                        _matrix->data[i][j] = 0;
                    }
                }
                if (type == CONSTANT) {
                    _matrix->data[i][j] = value;
                }
            }
        }
    }

    std::shared_ptr<Matrix<T, cols, rows>> generateMatrix()
    {
        _matrix->sequence = ++_sequence;
        _matrix->timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
        if (_type == RANDOM) {
            for (size_t i = 0; i < cols; i++) {
                for (size_t j = 0; j < rows; j++) {
                    _matrix->data[i][j] = _randomGenerator();
                }
            }
        }
        return _matrix;
    }

private:
    int _sequence;
    MatrixType _type;
    std::shared_ptr<Matrix<T, cols, rows>> _matrix;
    std::function<T()> _randomGenerator;
};
} // namespace matrix_mult_benchmark
} // namespace kpsr

#endif // MATRIX_DATA_FACTORY_H
