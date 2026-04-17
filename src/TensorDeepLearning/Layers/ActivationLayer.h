#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "Layer.h"
#include <ArenaTensorNode.h>
#include <GeneratorWrapper.h>

template <typename T> class ActivationLayer : public Layer<T> {
private:
  std::function<T(T &)> ActivationFunction;

public:
  ActivationLayer(std::function<T(T &)> ActivationFunction)
      : ActivationFunction(ActivationFunction) {}

  T Predict(T &Input, [[maybe_unused]] bool ComputeInputGradient = true) {
    return ActivationFunction(Input);
  }

  LayerType GetLayerType() const { return LayerType::Activation; }

  size_t ComputeParameterCount() { return 0; }

  void WriteLayerParameters([[maybe_unused]] ArenaArray<T *> &Array) {
    assert(false);
  }
};

#endif