#ifndef FIXED_RATE_SCHEDULER
#define FIXED_RATE_SCHEDULER

#include "Scheduler.h"

#include <Tensor.h>

template <typename T> class FixedRateScheduler : public Scheduler<T> {
public:
  FixedRateScheduler(T BaseLearningRate)
      : Scheduler<T>::Scheduler(BaseLearningRate) {}

  T GetRate([[maybe_unused]] size_t Steps) { return this->BaseLearningRate; }
};

#endif