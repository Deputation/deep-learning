#ifndef STOCHASTIC_GRADIENT_DESCENT_H
#define STOCHASTIC_GRADIENT_DESCENT_H

#include "Optimizer.h"

#include <ArenaArray.h>
#include <TensorOperations.h>

template <typename T> class AdaptiveGradient : public Optimizer<T> {
public:
  ArenaArray<Tensor<T>> SquaredGradientsSum;
  T Epsilon;

  AdaptiveGradient(ArenaArray<ArenaTensorNode<T> *> OptimizationParameters,
                   Scheduler<T> *LearningRateScheduler, T Epsilon = 1e-4)
      : Optimizer<T>::Optimizer(OptimizationParameters, LearningRateScheduler),
        SquaredGradientsSum(OptimizationParameters.Size), Epsilon(Epsilon) {
    for (size_t i = 0; i < SquaredGradientsSum.Size; i++) {
      SquaredGradientsSum[i] = Tensor<T>(
          OptimizationParameters[i]->GetValue().Dimensions.AllocateCopy(), 0);
    }
  }

  void Optimize() {
    Optimizer<T>::Optimize();

    auto LearningRate = this->LearningRateScheduler->GetRate(this->Steps);

    for (size_t i = 0; i < this->OptimizationParameters.Size; i++) {
      auto TensorNodeValue =
          this->OptimizationParameters[i]->GetTensorValuePointer();
      auto Gradient = TensorNodeValue->Gradient;
      T Exponent = 2;
      auto GradientSquared = Pow(Gradient, Exponent);
      auto GrowingGradient = SquaredGradientsSum[i] + GradientSquared;

      Copy(SquaredGradientsSum[i], GrowingGradient);

      auto GrowingGradientRoot = GrowingGradient + Epsilon;
      SqrtInPlace(GrowingGradientRoot);
      auto LearningRateTensor = Tensor<T>(
          GrowingGradientRoot.Dimensions.AllocateCopy(), LearningRate);
      LearningRateTensor /= GrowingGradientRoot;
      LearningRateTensor *= Gradient;

      TensorNodeValue->Value -= LearningRateTensor;
    }
  }
};

#endif
