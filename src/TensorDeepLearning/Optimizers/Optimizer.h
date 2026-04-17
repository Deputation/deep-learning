#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ArenaArray.h>
#include <ArenaTensorNode.h>

#include "Schedulers/Scheduler.h"

template <typename T> class Optimizer {
protected:
  ArenaArray<ArenaTensorNode<T> *> OptimizationParameters;

public:
  Scheduler<T> *LearningRateScheduler;
  size_t Steps;

  Optimizer(ArenaArray<ArenaTensorNode<T> *> OptimizationParameters,
            Scheduler<T> *LearningRateScheduler)
      : OptimizationParameters(OptimizationParameters),
        LearningRateScheduler(LearningRateScheduler), Steps(0) {
    assert(OptimizationParameters.Size > 1);
    assert(LearningRateScheduler != nullptr);
  }

  void ZeroGradients() {
    for (size_t i = 0; i < OptimizationParameters.Size; i++) {
      OptimizationParameters[i]->GetTensorValuePointer()->ZeroGradient();
    }
  }

  void Optimize() { Steps++; }
};

#endif