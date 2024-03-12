#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <klepsydra/core/yaml_environment.h>
#include <klepsydra/performance_benchmark/main_helper.h>

#include <klepsydra/matrix_mult_benchmark/matrix.h>
#include <klepsydra/performance_benchmark/configuration_data.h>
#include <klepsydra/performance_benchmark/file_admin_statistics_factory.h>

#include <klepsydra/matrix_mult_benchmark/matrix_seq_multiplier.h>
#include <klepsydra/matrix_mult_benchmark/stream_assembler.h>
#ifdef openmp_enabled
#include <klepsydra/matrix_mult_benchmark/matrix_openmp_multiplier.h>
#endif
#ifdef blas_enabled
#include <klepsydra/matrix_mult_benchmark/matrix_blas_multipler.h>
#endif
#ifdef eigen_enabled
#include <klepsydra/matrix_mult_benchmark/matrix_eigen_multiplier.h>
#endif
#ifdef ruy_enabled
#include <klepsydra/matrix_mult_benchmark/matrix_ruy_multiplier.h>
#endif

#include <klepsydra/mem_performance_benchmark/event_emitter_factory.h>
#include <klepsydra/mem_performance_benchmark/multi_event_loop_factory.h>

const int EVENT_LOOP_SIZE(256);
const int MATRIX_ROWS(100);

template<class T>
void matrixMutiplicationTest(kpsr::Environment *environment,
                             kpsr::performance_benchmark::ConfigurationData &configurationData,
                             std::function<T()> randomGenerator)
{
    spdlog::set_pattern("[%c] [%H:%M:%S %f] [%n] [%l] [%t] %v");
    spdlog::set_level(configurationData.toStdOut
                          ? spdlog::level::debug
                          : spdlog::level::info); // Set global log level to info

    if (configurationData.logToFile) {
        auto kpsrLogger = spdlog::basic_logger_mt("mem_matrix_multiplication_benchmark",
                                                  configurationData.logFilename);
        spdlog::set_default_logger(kpsrLogger);
    } else {
        auto kpsrLogger = spdlog::stdout_color_mt("mem_matrix_multiplication_benchmark");
        spdlog::set_default_logger(kpsrLogger);
    }

    if (configurationData.dataProcType == "kpsr_event_loop") {
        kpsr::Threadpool::getCriticalThreadPool(std::thread::hardware_concurrency() * 2 + 2);
        kpsr::Threadpool::getNonCriticalThreadPool(2);
    } else {
        kpsr::Threadpool::getCriticalThreadPool(1);
        kpsr::Threadpool::getNonCriticalThreadPool(2);
    }

    kpsr::performance_benchmark::FileAdminStatisticsFactory statisticsFactory(environment,
                                                                              configurationData);
    kpsr::Container *container = statisticsFactory.getContainer();

    kpsr::performance_benchmark::MultiEventLoopFactory<
        EVENT_LOOP_SIZE,
        kpsr::matrix_mult_benchmark::Matrix<T, MATRIX_ROWS, MATRIX_ROWS>> *eventLoopFactory = nullptr;
    kpsr::performance_benchmark::EventEmitterFactory<
        kpsr::matrix_mult_benchmark::Matrix<T, MATRIX_ROWS, MATRIX_ROWS>> *eventEmitterFactory =
        nullptr;

    spdlog::info("Creating streams....");
    kpsr::matrix_mult_benchmark::StreamAssembler<T, MATRIX_ROWS> *streamAssembler;
    std::vector<
        kpsr::matrix_mult_benchmark::MatrixMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS> *>
        matrixMultiplier;
    kpsr::matrix_mult_benchmark::MatrixDataFactory<T, MATRIX_ROWS, MATRIX_ROWS>
        matrixDataFactory((kpsr::matrix_mult_benchmark::MatrixType) configurationData.payloadSize,
                          10.0,
                          randomGenerator);

    if (configurationData.msgToSave == 0) {
        spdlog::info("MatrixSeqMultiplier....");
        matrixMultiplier.emplace_back(
            new kpsr::matrix_mult_benchmark::
                MatrixSeqMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>());
    } else if (configurationData.msgToSave == 1) {
#ifdef openmp_enabled
        spdlog::info("MatrixOmpMultiplier....");
        matrixMultiplier.emplace_back(
            new kpsr::matrix_mult_benchmark::
                MatrixOmpMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>());
#else
        spdlog::error("MatrixOmpMultiplier is disabled");
        return;
#endif
    } else if (configurationData.msgToSave == 2) {
#ifdef blas_enabled
        spdlog::info("MatrixBlasMultiplier....");
        matrixMultiplier.emplace_back(
            new kpsr::matrix_mult_benchmark::
                MatrixBlasMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>());
#else
        spdlog::error("MatrixBlasMultiplier is disabled");
        return;
#endif
    } else if (configurationData.msgToSave == 3) {
#ifdef eigen_enabled
        spdlog::info("MatrixTemplatedEigenMultiplier....");
        matrixMultiplier.emplace_back(
            new kpsr::matrix_mult_benchmark::
                MatrixTemplatedEigenMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>());
#else
        spdlog::error("MatrixTemplatedEigenMultiplier is disabled");
        return;
#endif
    } else if (configurationData.msgToSave == 4) {
#ifdef ruy_enabled
        spdlog::info("MatrixRuyMultiplier.....");
        for (int i = 0; i < configurationData.topicCount; i++) {
            matrixMultiplier.emplace_back(
                new kpsr::matrix_mult_benchmark::
                    MatrixRuyMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>());
        }
#else
        spdlog::error("MatrixRuyMultiplier is disabled");
        return;
#endif
    }
    if (configurationData.dataProcType == "kpsr_event_loop") {
        eventLoopFactory = new kpsr::performance_benchmark::MultiEventLoopFactory<
            EVENT_LOOP_SIZE,
            kpsr::matrix_mult_benchmark::Matrix<T, MATRIX_ROWS, MATRIX_ROWS>>(
            configurationData.topicCount,
            configurationData.eventPoolSize,
            configurationData.topicPrefix,
            container);
        streamAssembler = new kpsr::matrix_mult_benchmark::StreamAssembler<T, MATRIX_ROWS>(
            configurationData.topicPrefix,
            configurationData.topicCount,
            configurationData.publishingRate,
            eventLoopFactory,
            eventLoopFactory,
            eventLoopFactory,
            matrixMultiplier,
            &matrixDataFactory,
            configurationData.toStdOut,
            false);
    } else {
        bool sequential = (configurationData.dataProcType != "kpsr_event_emitter");
        eventEmitterFactory = new kpsr::performance_benchmark::EventEmitterFactory<
            kpsr::matrix_mult_benchmark::Matrix<T, MATRIX_ROWS, MATRIX_ROWS>>(
            configurationData.topicCount,
            configurationData.eventPoolSize,
            configurationData.topicPrefix,
            container);
        streamAssembler = new kpsr::matrix_mult_benchmark::StreamAssembler<T, MATRIX_ROWS>(
            configurationData.topicPrefix,
            configurationData.topicCount,
            configurationData.publishingRate,
            eventEmitterFactory,
            eventEmitterFactory,
            eventEmitterFactory,
            matrixMultiplier,
            &matrixDataFactory,
            configurationData.toStdOut,
            sequential);
    }

    spdlog::info("starting....");
    std::this_thread::sleep_for(std::chrono::milliseconds(configurationData.testDelayInMs));

    if (eventLoopFactory != nullptr) {
        eventLoopFactory->start();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(configurationData.testDelayInMs));

    statisticsFactory.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(configurationData.testDelayInMs));

    streamAssembler->start();

    spdlog::info("running....");
    std::this_thread::sleep_for(std::chrono::seconds(configurationData.testDuration));

    spdlog::info("stopping....");
    streamAssembler->stop();

    statisticsFactory.stop();

    spdlog::info("Total processed matrices: {}", streamAssembler->totalProcessedMatrices);
    spdlog::info("Total processing time: {}", streamAssembler->totalProcessingTime);

    if (eventLoopFactory != nullptr) {
        eventLoopFactory->stop();
    }

    delete streamAssembler;

    if (eventLoopFactory) {
        delete eventLoopFactory;
    }
    if (eventEmitterFactory) {
        delete eventEmitterFactory;
    }

#ifdef eigen_enabled
    if (configurationData.msgToSave == 3) {
        spdlog::info(
            "MatrixTemplatedEigenMultiplier performed {} multiplications",
            static_cast<kpsr::matrix_mult_benchmark::
                            MatrixTemplatedEigenMultiplier<T, MATRIX_ROWS, MATRIX_ROWS, MATRIX_ROWS>
                                *>(matrixMultiplier[0])
                ->numberOfMultiplications);
    }
#endif
    spdlog::info("finished....");
}

int main(int argc, char **argv)
{
    std::string filename;
    kpsr::performance_benchmark::MainHelper::getConfFileFromParams(argc, argv, filename);
    kpsr::YamlEnvironment environment(filename);
    kpsr::performance_benchmark::ConfigurationData configurationData(&environment);

    if (configurationData.jsonDir == "double") {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 10.0);
        matrixMutiplicationTest<double>(&environment, configurationData, [&]() {
            return dist(gen);
        });
        return 0;
    }
    if (configurationData.jsonDir == "float") {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 10.0);
        matrixMutiplicationTest<float>(&environment, configurationData, [&]() { return dist(gen); });
        return 0;
    }
    if (configurationData.jsonDir == "integer") {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, 10);
        matrixMutiplicationTest<int>(&environment, configurationData, [&]() { return dist(gen); });
        return 0;
    }
}
