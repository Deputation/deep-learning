#ifndef ADAM_H
#define ADAM_H

#include "Optimizer.h"

#include <ArenaArray.h>
#include <ArenaTensorNode.h>

template <typename T> class Adam : public Optimizer<T> {
public:
  ArenaArray<Tensor<T>> Momentum;
  ArenaArray<Tensor<T>> SecondMoment;

  T MomentumFactor;
  T SecondMomentFactor;

  T Epsilon;

  Adam(ArenaArray<ArenaTensorNode<T> *> OptimizationParameters,
       Scheduler<T> *LearningRateScheduler, T MomentumFactor,
       T SecondMomentFactor, T Epsilon)
      : Optimizer<T>::Optimizer(OptimizationParameters, LearningRateScheduler),
        Momentum(OptimizationParameters.Size),
        SecondMoment(OptimizationParameters.Size),
        MomentumFactor(MomentumFactor), SecondMomentFactor(SecondMomentFactor),
        Epsilon(Epsilon) {
    for (size_t i = 0; i < Momentum.Size; i++) {
      Momentum[i] = Tensor<T>(OptimizationParameters[i]
                                  ->GetTensorValuePointer()
                                  ->Value.Dimensions.AllocateCopy(),
                              0);
    }
    for (size_t i = 0; i < SecondMoment.Size; i++) {
      SecondMoment[i] = Tensor<T>(OptimizationParameters[i]
                                      ->GetTensorValuePointer()
                                      ->Value.Dimensions.AllocateCopy(),
                                  0);
    }
  }

  void Optimize() {
    Optimizer<T>::Optimize();

    auto LearningRate = this->LearningRateScheduler->GetRate(this->Steps);

    for (size_t i = 0; i < this->OptimizationParameters.Size; i++) {
      auto TensorNodeValue =
          this->OptimizationParameters[i]->GetTensorValuePointer();
      auto Gradient = TensorNodeValue->Gradient;

      auto MomentumUpdate =
          (Momentum[i] * MomentumFactor) + (Gradient * (1 - MomentumFactor));
      Copy(Momentum[i], MomentumUpdate);

      auto SecondMomentUpdate =
          (SecondMoment[i] * SecondMomentFactor) +
          ((Gradient * Gradient) * (1 - SecondMomentFactor));
      Copy(SecondMoment[i], SecondMomentUpdate);

      auto NormalizedMomentum = Momentum[i] / (1 - MomentumFactor);
      auto NormalizedSecondMoment = SecondMoment[i] / (1 - SecondMomentFactor);

      auto UpdateValue = (NormalizedMomentum * LearningRate) /
                         (Sqrt(NormalizedSecondMoment) + Epsilon);

      TensorNodeValue->Value -= UpdateValue;
    }
  }
};

#endif