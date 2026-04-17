#ifndef MODEL_H
#define MODEL_H

#include "Layers/Layer.h"
#include "Tensor.h"
#include <ArenaArray.h>
#include <fstream>

template <typename T> class Model {
private:
  ArenaArray<Layer<T> *> Layers;

  size_t ComputeTotalParameterCount() {
    size_t Size = 0;
    for (size_t i = 0; i < Layers.Size; i++) {
      Size += Layers[i]->ComputeParameterCount();
    }
    return Size;
  }

  ArenaArray<T *> ComputeOptimizationParametersArray() {
    ArenaArray<T *> Parameters(ComputeTotalParameterCount());

    for (size_t i = 0; i < Layers.Size; i++) {
      if (Layers[i]->ComputeParameterCount() > 0) {
        Layers[i]->WriteLayerParameters(Parameters);
      }
    }

    return Parameters;
  }

public:
  ArenaArray<T *> OptimizationParameters;

  Model(ArenaArray<Layer<T> *> Layers) : Layers(Layers) {
    OptimizationParameters = ComputeOptimizationParametersArray();
  }

  T Predict(T &Input, bool ComputeInputGradients = false) {
    auto Result = Layers[0]->Predict(Input, ComputeInputGradients);

    for (size_t i = 1; i < Layers.Size; i++) {
      Result = Layers[i]->Predict(Result, true);
    }

    return Result;
  }

  bool WriteToDisk(const std::string &Path) {
    std::ofstream File(Path, std::ios::binary);

    if (!File.good()) {
      assert(false);
      return false;
    }

    for (size_t i = 0; i < OptimizationParameters.Size; i++) {
      void *TensorObject = nullptr;
      if constexpr (IsArenaTensorNode<T>::value) {
        TensorObject = &OptimizationParameters[i]->NodePointer->Value;
      } else if constexpr (IsTensor<T>::value) {
        TensorObject = OptimizationParameters[i];
      } else {
        assert(false);
      }

      assert(TensorObject != nullptr);

      if constexpr (IsArenaTensorNode<T>::value) {
        auto CastedTensorObject =
            (Tensor<typename T::ArenaTensorNodeType> *)TensorObject;

        for (size_t j = 0; j < CastedTensorObject->Data.Size; j++) {
          File << CastedTensorObject->Data[j];
        }
      } else if constexpr (IsTensor<T>::value) {
        auto CastedTensorObject =
            (Tensor<typename T::TensorElementType> *)TensorObject;

        for (size_t j = 0; j < CastedTensorObject->Data.Size; j++) {
          File << CastedTensorObject->Data[j];
        }
      }
    }

    File.close();

    return true;
  }

  bool LoadFromDisk(const std::string &Path) {
    std::ifstream File(Path, std::ios::binary);

    if (!File.good()) {
      assert(false);
      return false;
    }

    for (size_t i = 0; i < OptimizationParameters.Size; i++) {
      void *TensorObject = nullptr;
      if constexpr (IsArenaTensorNode<T>::value) {
        TensorObject = &OptimizationParameters[i]->NodePointer->Value;
      } else if constexpr (IsTensor<T>::value) {
        TensorObject = OptimizationParameters[i];
      } else {
        assert(false);
      }

      assert(TensorObject != nullptr);

      if constexpr (IsArenaTensorNode<T>::value) {
        auto CastedTensorObject =
            (Tensor<typename T::ArenaTensorNodeType> *)TensorObject;

        for (size_t j = 0; j < CastedTensorObject->Data.Size; j++) {
          File >> CastedTensorObject->Data[j];
        }
      } else if constexpr (IsTensor<T>::value) {
        auto CastedTensorObject =
            (Tensor<typename T::TensorElementType> *)TensorObject;

        for (size_t j = 0; j < CastedTensorObject->Data.Size; j++) {
          File >> CastedTensorObject->Data[j];
        }
      }
    }

    File.close();

    return true;
  }
};

#endif