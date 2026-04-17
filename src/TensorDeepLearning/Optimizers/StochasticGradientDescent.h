#ifndef STOCHASTIC_GRADIENT_DESCENT_H
#define STOCHASTIC_GRADIENT_DESCENT_H

#include "Optimizer.h"

#include <ArenaArray.h>

template <typename T> class StochasticGradientDescent : public Optimizer<T> {
public:
  ArenaArray<Tensor<T>> Momentum;
  T MomentumFactor;
  T WeightDecayFactor;

  StochasticGradientDescent(
      ArenaArray<ArenaTensorNode<T> *> OptimizationParameters,
      Scheduler<T> *LearningRateScheduler, T MomentumFactor = 0.9,
      T WeightDecayFactor = 0)
      : Optimizer<T>::Optimizer(OptimizationParameters, LearningRateScheduler),
        Momentum(OptimizationParameters.Size), MomentumFactor(MomentumFactor),
        WeightDecayFactor(WeightDecayFactor) {
    for (size_t i = 0; i < Momentum.Size; i++) {
      Momentum[i] = Tensor<T>(
          OptimizationParameters[i]->GetValue().Dimensions.AllocateCopy(), 0);
    }
  }

  void Optimize() {
    Optimizer<T>::Optimize();

    auto LearningRate = this->LearningRateScheduler->GetRate(this->Steps);

    for (size_t i = 0; i < this->OptimizationParameters.Size; i++) {
      auto TensorNodeValue =
          this->OptimizationParameters[i]->GetTensorValuePointer();
      auto Gradient = TensorNodeValue->Gradient +
                      (TensorNodeValue->Value * WeightDecayFactor);

      auto UpdateValue =
          (Momentum[i] * MomentumFactor) + (Gradient * (1 - MomentumFactor));

      Copy(Momentum[i], UpdateValue);

      auto SubtractionValue = UpdateValue * LearningRate;
      TensorNodeValue->Value -= SubtractionValue;
    }
  }
};

#endif
