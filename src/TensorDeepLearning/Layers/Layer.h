#ifndef LAYER_H
#define LAYER_H

#include <ArenaTensorNode.h>

enum class LayerType : unsigned short { Linear, Activation };

template <typename T> class Layer {
public:
  virtual T Predict(T &Input, bool ComputeInputGradient = false) = 0;
  virtual LayerType GetLayerType() const = 0;
  virtual size_t ComputeParameterCount() = 0;
  virtual void WriteLayerParameters(ArenaArray<T *> &Array) = 0;
};

#endif