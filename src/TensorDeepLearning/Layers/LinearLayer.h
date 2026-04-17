#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "Layer.h"
#include "Tensor.h"

#include <ArenaTensorNode.h>
#include <GeneratorWrapper.h>

template <typename T> class LinearLayer : public Layer<T> {
public:
  T WeightMatrix;
  T BiasVector;

  LinearLayer(size_t InputSize, size_t OutputSize)
    requires IsArenaTensorNode<T>::value
      : WeightMatrix(Tensor<typename T::ArenaTensorNodeType>(
            ArenaArray<size_t>({OutputSize, InputSize}),
            []() {
              return GeneratorWrapper<
                         typename T::ArenaTensorNodeType>::GetSingleton()
                  .Sample();
            })),
        BiasVector(Tensor<typename T::ArenaTensorNodeType>(
            ArenaArray<size_t>({OutputSize, 1}), []() {
              return GeneratorWrapper<
                         typename T::ArenaTensorNodeType>::GetSingleton()
                  .Sample();
            })) {}

  LinearLayer(size_t InputSize, size_t OutputSize)
    requires IsTensor<T>::value
      : WeightMatrix(Tensor<typename T::TensorElementType>(
            ArenaArray<size_t>({OutputSize, InputSize}),
            []() {
              return GeneratorWrapper<
                         typename T::TensorElementType>::GetSingleton()
                  .Sample();
            })),
        BiasVector(Tensor<typename T::TensorElementType>(
            ArenaArray<size_t>({OutputSize, 1}), []() {
              return GeneratorWrapper<
                         typename T::TensorElementType>::GetSingleton()
                  .Sample();
            })) {}

  size_t GetColumnDimension(T &Value) {
    if constexpr (IsArenaTensorNode<T>::value) {
      return Value.NodePointer->Value.Dimensions[1];
    } else if constexpr (IsTensor<T>::value) {
      return Value.Dimensions[1];
    } else {
      assert(false);
      return -1;
    }
  }

  T ComputeExtendedBias(T &Result) {
    size_t NewColumns = GetColumnDimension(Result);
    return ExtendVector(BiasVector, 1, NewColumns);
  }

  T Predict(T &Input, bool ComputeInputGradient = false) {
    T Result;

    if constexpr (IsArenaTensorNode<T>::value) {
      if (!ComputeInputGradient) {
        Result = MatrixMultiplication(WeightMatrix, Input.NodePointer->Value);
      } else {
        Result = MatrixMultiplication(WeightMatrix, Input);
      }
    } else if constexpr (IsTensor<T>::value) {
      Result = MatrixMultiplication(WeightMatrix, Input);
    }

    if (GetColumnDimension(Result) > 1) {
      auto ExtendedBias = ComputeExtendedBias(Result);
      return Result + ExtendedBias;
    }

    return Result + BiasVector;
  }

  LayerType GetLayerType() const { return LayerType::Linear; }

  size_t ComputeParameterCount() { return 2; }

  void WriteLayerParameters(ArenaArray<T *> &Array) {
    Array.Push(&WeightMatrix);
    Array.Push(&BiasVector);
  }
};

#endif