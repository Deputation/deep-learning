#ifndef TENSOR_UTILITIES_H
#define TENSOR_UTILITIES_H

#include "Tensor.h"
#include <iostream>

template <typename T> static void PrintStrides(Tensor<T> &Input) {
  std::cout << "Strides: ";
  for (size_t i = 0; i < Input.Strides.Size; i++) {
    std::cout << Input.Strides[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T> static void PrintDimensions(Tensor<T> &Input) {
  std::cout << "Dimensions: ";
  for (size_t i = 0; i < Input.Dimensions.Size; i++) {
    std::cout << Input.Dimensions[i] << " ";
  }
  std::cout << std::endl;
}

template <typename T> static void PrintAsMatrix(Tensor<T> &Input) {
  for (size_t i = 0; i < Input.Dimensions[0]; i++) {
    for (size_t j = 0; j < Input.Dimensions[1]; j++) {
      std::cout << Input[i, j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T> static void Print(Tensor<T> &Input) {
  if (Input.Type == TensorType::Scalar) {
    std::cout << Input[0] << std::endl;
    return;
  }

  PrintDimensions(Input);
  PrintStrides(Input);

  if (Input.Type <= TensorType::Matrix) {
    PrintAsMatrix(Input);
    return;
  }

  assert(false);
}

#endif