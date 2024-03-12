#include <iostream>

#include <chrono>
#include <thread>
#include <vector>

#include <algorithm>
#include <random>

#include <functional>
#include <iostream>

#include "matrix.h"
#include "matrix_multiplier.h"
#include "matrix_seq_multiplier.h"
#include "matrix_openmp_multiplier.h"
#include "matrix_eigen_multiplier.h"

constexpr int MATRIX_ROWS = 100;

using T = float;

template <class TimeUnit>
class TimingInfos {
public:
    TimingInfos() : timeDiff() {}
       
    void start() {
        startTime = std::chrono::steady_clock::now();
    }

    void stop() {
        timeDiff += std::chrono::duration_cast<TimeUnit>(
            std::chrono::steady_clock::now() - startTime).count();
    }
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    typename TimeUnit::rep timeDiff;
};

int main() {

    using Mat = Matrix<T, MATRIX_ROWS, MATRIX_ROWS>;
    using MatMul = MatrixMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>;
    std::unique_ptr<MatMul> matrixMultiplier;
    
    constexpr int numCores = 4;
    constexpr int numIterations = 20;
    using TimeUnit = std::chrono::milliseconds;
    using TimingType = TimingInfos<TimeUnit>;

    auto runBenchmarks = [numCores, numIterations](MatMul *matrixMultiplier) {
        // fill input matrices
        std::vector<Mat>  inputMatrices;
        std::vector<Mat>  outputMatrices;

    
        std::random_device random_device;
        auto rng = std::mt19937(random_device());
        auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));

        for (int i = 0; i < numCores; i++) {
            inputMatrices.push_back(Mat(i));
            outputMatrices.push_back(Mat(i));
            auto &mat = inputMatrices.back();
            std::generate(&mat.data[0][0], &mat.data[MATRIX_ROWS-1][MATRIX_ROWS-1], std::ref(f32rng));
        }

        std::vector<TimingType> timings(inputMatrices.size());
        auto matMulFunction = [&inputMatrices, &outputMatrices, &timings, numIterations](int i, MatMul *matrixMultiplier) {
            for (int k = 0; k < numIterations; k++) {
                timings[i].start();
                matrixMultiplier->multiply(inputMatrices[i], inputMatrices[i], outputMatrices[i]);
                timings[i].stop();
            }
        };

        TimingType singleTime;
        auto singleCall = [&inputMatrices, &outputMatrices, &singleTime, numIterations](MatMul *matrixMultiplier){
            singleTime.start();
            for (int k = 0; k < numIterations; k++) {
                for (int i = 0; i < inputMatrices.size(); i++) {
                    matrixMultiplier->multiply(inputMatrices[i], inputMatrices[i], outputMatrices[i]);
                }
            }
            singleTime.stop();
        };
        std::vector<std::thread> parallelThreads;
        TimingType threadedTime;
        threadedTime.start();
        for (int i = 0; i < numCores; i++) {
            parallelThreads.emplace_back(std::thread(matMulFunction, i, matrixMultiplier));
        }

        for (auto &t: parallelThreads) {
            t.join();
        }
        threadedTime.stop();

        singleCall(matrixMultiplier);
        decltype(TimingType::timeDiff) separateThreadTimes = 0;
        std::cout << "Running each layers for " << numIterations << " iterations " << std::endl;
        for (size_t  i = 0; i < numCores; i++) {
            separateThreadTimes += timings[i].timeDiff;
            // std::cout << "Matrix mult " << i << " executed in " << timings[i].timeDiff << " us " << std::endl;
        }

        std::cout << "Separate thread timings : " << separateThreadTimes << std::endl;
        std::cout << "Separate thread  total time : " << threadedTime.timeDiff << std::endl;
        std::cout <<"Sequential Thread timings : " << singleTime.timeDiff << std::endl;
    };

    matrixMultiplier.reset(nullptr);
#ifdef openmp_enabled
    matrixMultiplier.reset(nullptr);
    std::cout << "Tests run with openmp_enabled " << std::endl;
    matrixMultiplier = std::make_unique<MatrixOmpMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>>();
    runBenchmarks(matrixMultiplier.get());
#endif
#if eigen_enabled
    matrixMultiplier.reset(nullptr);
    std::cout << "Tests run with eigen_enabled " << std::endl;
    matrixMultiplier = std::make_unique<MatrixTemplatedEigenMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>>();
    runBenchmarks(matrixMultiplier.get());    
#endif
    std::cout << "Tests run with normal mode " << std::endl;
    matrixMultiplier.reset(nullptr);
    matrixMultiplier = std::make_unique<MatrixSeqMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>>();
    runBenchmarks(matrixMultiplier.get());    

}
