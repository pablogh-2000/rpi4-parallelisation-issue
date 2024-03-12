#ifndef MM4CONVKN_STREAM_ASSEMBLER_H
#define MM4CONVKN_STREAM_ASSEMBLER_H

#include <iostream>
#include <vector>

#include <klepsydra/core/event_transform_forwarder.h>

#include <klepsydra/performance_benchmark/publisher_factory.h>
#include <klepsydra/performance_benchmark/scheduler_factory.h>
#include <klepsydra/performance_benchmark/subscriber_factory.h>

#include <klepsydra/matrix_mult_benchmark/matrix_multiplier.h>

#include <klepsydra/matrix_mult_benchmark/matrix_operation_event.h>

namespace kpsr {
namespace matrix_mult_benchmark {

template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
class ConvolutionKNMockTransformForwarder
{
public:
    ConvolutionKNMockTransformForwarder(
        MatrixOperationEvent<T, colsLeft, rowsRight> &matrixOperationEvent,
        Subscriber<unsigned long> *previousSubscriber,
        Publisher<unsigned long> *nextPublisher,
        MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight> *matrixMultiplier)
        : eventTranformForwarder(
              [matrixMultiplier, this](const unsigned long preSeq, unsigned long &nextSeq) {
                  Matrix<T, colsLeft, rowsRight> C;
                  for (int i = 0; i < _matrixOperationEvent.repetitions; i++) {
                      matrixMultiplier->multiply(*_matrixOperationEvent.left,
                                                 *_matrixOperationEvent.right,
                                                 C);
                  }
                  nextSeq = preSeq;
              },
              nextPublisher)
        , _previousSubscriber(previousSubscriber)
        , _matrixOperationEvent(matrixOperationEvent)
    {
        previousSubscriber->registerListener("ConvolutionKNMockTransformForwarder",
                                             eventTranformForwarder.forwarderListenerFunction);
    }

    virtual ~ConvolutionKNMockTransformForwarder()
    {
        _previousSubscriber->removeListener("ConvolutionKNMockTransformForwarder");
    }

private:
    EventTransformForwarder<unsigned long, unsigned long> eventTranformForwarder;
    Subscriber<unsigned long> *_previousSubscriber;
    MatrixOperationEvent<T, colsLeft, rowsRight> &_matrixOperationEvent;
};

class MM4ConvKnStreamAssembler
{
public:
    MM4ConvKnStreamAssembler(const std::string &providerNamePrefix,
                             int period,
                             SchedulerFactory *schedulerFactory,
                             SubscriberFactory<unsigned long> *subscriberFactory,
                             PublisherFactory<unsigned long> *producerFactory)
        : _period(period)
        , _schedulerFactory(schedulerFactory)
        , _subscriberFactory(subscriberFactory)
        , _producerFactory(producerFactory)
        , _providerNamePrefix(providerNamePrefix)
        , _counter(0)
    {}

    template<class T, size_t colsLeft, size_t rowsLeft, size_t rowsRight>
    std::shared_ptr<ConvolutionKNMockTransformForwarder<T, colsLeft, rowsLeft, rowsRight>>
    addConvLayer(MatrixOperationEvent<T, colsLeft, rowsRight> &event,
                 MatrixMultiplier<T, colsLeft, rowsLeft, rowsRight> *matrixMultiplier)
    {
        std::string const previousProviderName = _providerNamePrefix + std::to_string(_counter);
        std::string const nextProviderName = _providerNamePrefix + std::to_string(_counter + 1);
        std::shared_ptr<std::function<void()>> producerFunction =
            std::make_shared<std::function<void()>>([previousProviderName, this]() {
                unsigned long currentTimetamp =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
                _producerFactory->getPublisher(previousProviderName)->publish(currentTimetamp);
            });

        _producerFunctions.push_back((producerFunction));

        kpsr::Subscriber<unsigned long> *previousSubscriber = _subscriberFactory->getSubscriber(
            previousProviderName);
        kpsr::Publisher<unsigned long> *nextPublisher = _producerFactory->getPublisher(
            nextProviderName);
        auto stream =
            std::make_shared<ConvolutionKNMockTransformForwarder<T, colsLeft, rowsLeft, rowsRight>>(
                event, previousSubscriber, nextPublisher, matrixMultiplier);

        _counter++;

        return stream;
    }

    void start()
    {
        std::string const lastProviderName = _providerNamePrefix + std::to_string(_counter);
        _subscriberFactory->getSubscriber(lastProviderName)
            ->registerListener("StreamAssembler", [&](const unsigned long &id) {
                long long currentTimetamp = std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count();
                totalProcessingTime = totalProcessingTime + (currentTimetamp - id);
                totalProcessedMatrices++;
            });

        for (size_t i = 0; i < _producerFunctions.size(); i++) {
            _schedulerFactory->getScheduler()->startScheduledTask(std::to_string(i),
                                                                  _period * 1000,
                                                                  true,
                                                                  _producerFunctions[i]);
        }
    }

    void stop()
    {
        for (size_t i = 0; i < _producerFunctions.size(); i++) {
            _schedulerFactory->getScheduler()->stopScheduledTask(std::to_string(i));
        }
    }

    virtual ~MM4ConvKnStreamAssembler()
    {
        std::string const lastProviderName = _providerNamePrefix + std::to_string(_counter);
        _subscriberFactory->getSubscriber(lastProviderName)->removeListener("StreamAssembler");
        _producerFunctions.clear();
    }

    long long totalProcessingTime = 0;
    int totalProcessedMatrices = 0;

private:
    int _period;
    SchedulerFactory *_schedulerFactory;
    SubscriberFactory<unsigned long> *_subscriberFactory;
    PublisherFactory<unsigned long> *_producerFactory;
    std::string _providerNamePrefix;
    int _counter;
    std::vector<std::shared_ptr<std::function<void()>>> _producerFunctions;
};
} // namespace matrix_mult_benchmark
} // namespace kpsr

#endif // MM4CONVKN_STREAM_ASSEMBLER_H
