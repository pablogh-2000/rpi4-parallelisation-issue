#ifndef MATRIX_H
#define MATRIX_H

#include <chrono>
#include <cstdio>

template<class T, size_t cols, size_t rows>
class Matrix
{
public:
    Matrix()
        : sequence(0)
    {}

    Matrix(int sequence)
        : sequence(sequence)
        , timestamp(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count())
    {}

    void row(int row, T *rowData)
    {
        for (int i = 0; i < cols; i++) {
            rowData[i] = data[row][i];
        }
    }

    int sequence;
    long long timestamp;
    T data[cols][rows];
};


#endif // MATRIX_H
