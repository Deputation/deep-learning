#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <Tensor.h>
#include <cstddef>

template <typename T> class Scheduler {
protected:
  T BaseLearningRate;

public:
  Scheduler(T BaseLearningRate) : BaseLearningRate(BaseLearningRate) {}

  virtual T GetRate(size_t Steps) = 0;
};

#endif