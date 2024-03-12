#ifndef STREAM_ASSEMBLER_H
#define STREAM_ASSEMBLER_H

#include <iostream>

#include <klepsydra/core/event_transform_forwarder.h>

#include <klepsydra/performance_benchmark/publisher_factory.h>
#include <klepsydra/performance_benchmark/scheduler_factory.h>
#include <klepsydra/performance_benchmark/subscriber_factory.h>

#include <klepsydra/matrix_mult_benchmark/matrix_data_factory.h>
#include <klepsydra/matrix_mult_benchmark/matrix_multiplier.h>

namespace kpsr {
namespace matrix_mult_benchmark {

template<class T, size_t rows>
class MatrixMultiplierTransformForwarder
{
public:
    MatrixMultiplierTransformForwarder(Subscriber<Matrix<T, rows, rows>> *previousSubscriber,
                                       Publisher<Matrix<T, rows, rows>> *nextPublisher,
                                       MatrixMultiplier<T, rows, rows, rows> *matrixMultiplier,
                                       bool debug)
        : eventTranformForwarder(
              [matrixMultiplier, debug](const Matrix<T, rows, rows> &A, Matrix<T, rows, rows> &C) {
                  matrixMultiplier->multiply(A, A, C);
                  if (debug) {
                      for (size_t i = 0; i < rows; i++) {
                          for (size_t j = 0; j < rows; j++) {
                              std::cout << C.data[i][j] << "\t";
                          }
                          std::cout << std::endl;
                      }
                  }
              },
              nextPublisher)
        , _previousSubscriber(previousSubscriber)
    {
        previousSubscriber->registerListener("MatrixMultiplierTransformForwarder",
                                             eventTranformForwarder.forwarderListenerFunction);
    }

    virtual ~MatrixMultiplierTransformForwarder()
    {
        _previousSubscriber->removeListener("MatrixMultiplierTransformForwarder");
    }

private:
    EventTransformForwarder<Matrix<T, rows, rows>, Matrix<T, rows, rows>> eventTranformForwarder;
    Subscriber<Matrix<T, rows, rows>> *_previousSubscriber;
};

template<class T, size_t rows>
class StreamAssembler
{
public:
    StreamAssembler(const std::string &providerNamePrefix,
                    const int topicCount,
                    int period,
                    SchedulerFactory *schedulerFactory,
                    SubscriberFactory<Matrix<T, rows, rows>> *subscriberFactory,
                    PublisherFactory<Matrix<T, rows, rows>> *producerFactory,
                    std::vector<MatrixMultiplier<T, rows, rows, rows> *> matrixMultiplier,
                    MatrixDataFactory<T, rows, rows> *matrixDataFactory,
                    bool debug,
                    bool sequential)
        : _period(period)
        , _streams(topicCount)
        , _schedulerFactory(schedulerFactory)
        , _subscriberFactory(subscriberFactory)
        , _providerNamePrefix(providerNamePrefix)
        , _topicCount(topicCount)
    {
        _matrixProducer = std::make_shared<std::function<void()>>(
            [producerFactory, providerNamePrefix, matrixDataFactory]() {
                std::string const previousProviderName = providerNamePrefix + "0";
                producerFactory->getPublisher(previousProviderName)
                    ->publish(matrixDataFactory->generateMatrix());
            });

        if (sequential) {
            std::string const previousProviderName = providerNamePrefix + "0";
            auto multiplier = matrixMultiplier[0];
            _subscriberFactory->getSubscriber(previousProviderName)
                ->registerListener("Multiplications",
                                   [&, multiplier, debug](const Matrix<T, rows, rows> &matrix) {
                                       Matrix<T, rows, rows> input = matrix;
                                       Matrix<T, rows, rows> output;
                                       for (int i = 0; i < (_topicCount - 1); i++) {
                                           multiplier->multiply(input, input, output);
                                           input = output;
                                       }
                                       if (debug) {
                                           for (size_t i = 0; i < rows; i++) {
                                               for (size_t j = 0; j < rows; j++) {
                                                   std::cout << output.data[i][j] << "\t";
                                               }
                                               std::cout << std::endl;
                                           }
                                       }
                                   });

        } else {
            for (int i = 0; i < (topicCount - 1); i++) {
                auto multiplier = matrixMultiplier.size() == 1 ? matrixMultiplier[0]
                                                               : matrixMultiplier[i];
                std::string const previousProviderName = providerNamePrefix + std::to_string(i);
                std::string const nextProviderName = providerNamePrefix + std::to_string(i + 1);
                kpsr::Subscriber<Matrix<T, rows, rows>> *previousSubscriber =
                    subscriberFactory->getSubscriber(previousProviderName);
                kpsr::Publisher<Matrix<T, rows, rows>> *nextPublisher =
                    producerFactory->getPublisher(nextProviderName);
                auto stream = std::make_shared<MatrixMultiplierTransformForwarder<T, rows>>(
                    previousSubscriber, nextPublisher, multiplier, debug);
                _streams[i] = stream;
            }
            std::string const previousProviderName = providerNamePrefix +
                                                     std::to_string(topicCount - 1);
            subscriberFactory->getSubscriber(previousProviderName)
                ->registerListener("StreamAssembler", [&](const Matrix<T, rows, rows> &matrix) {
                    long long currentTimetamp =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
                    totalProcessingTime = totalProcessingTime +
                                          (currentTimetamp - matrix.timestamp);
                    totalProcessedMatrices++;
                });
        }
    }

    void start()
    {
        _schedulerFactory->getScheduler()->startScheduledTask("StreamAssembler",
                                                              _period * 1000,
                                                              true,
                                                              _matrixProducer);
    }

    void stop() { _schedulerFactory->getScheduler()->stopScheduledTask("StreamAssembler"); }

    ~StreamAssembler()
    {
        std::string const previousProviderName = _providerNamePrefix +
                                                 std::to_string(_topicCount - 1);
        _subscriberFactory->getSubscriber(previousProviderName)->removeListener("StreamAssembler");
        _streams.clear();
    }

    long long totalProcessingTime = 0;
    int totalProcessedMatrices = 0;

private:
    int _period;
    std::vector<std::shared_ptr<MatrixMultiplierTransformForwarder<T, rows>>> _streams;
    SchedulerFactory *_schedulerFactory;
    SubscriberFactory<Matrix<T, rows, rows>> *_subscriberFactory;
    std::shared_ptr<std::function<void()>> _matrixProducer;
    std::string _providerNamePrefix;
    int _topicCount;
};
} // namespace matrix_mult_benchmark
} // namespace kpsr

#endif // STREAM_ASSEMBLER_H
