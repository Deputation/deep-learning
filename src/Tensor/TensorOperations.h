#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "Tensor.h"
#include <ArenaArray.h>
#include <string.h>

#define TensorOperatorOverload(OperatorName, Operator)                         \
  template <typename T>                                                        \
  Tensor<T> OperatorName(const Tensor<T> &Left, const T Right) {               \
    ArenaArray<T> Data(Left.Data.Size);                                        \
    Tensor<T> Result(Left.Dimensions.AllocateCopy(), Data);                    \
    for (size_t i = 0; i < Left.Data.Size; i++) {                              \
      Result[i] = Left[i] Operator Right;                                      \
    }                                                                          \
    return Result;                                                             \
  }                                                                            \
  template <typename T>                                                        \
  Tensor<T> OperatorName(const Tensor<T> &Left, const Tensor<T> &Right) {      \
    ArenaArray<T> Data(Left.Data.Size);                                        \
    Tensor<T> Result(Left.Dimensions.AllocateCopy(), Data);                    \
    assert(Left.DimensionsMatch(Right));                                       \
    assert(Left.Data.Size == Right.Data.Size);                                 \
    for (size_t i = 0; i < Left.Data.Size; i++) {                              \
      Result[i] = Left[i] Operator Right[i];                                   \
    }                                                                          \
    return Result;                                                             \
  }

TensorOperatorOverload(operator+, +);
TensorOperatorOverload(operator-, -);
TensorOperatorOverload(operator*, *);
TensorOperatorOverload(operator/, /);

template <typename T> Tensor<T> operator-(const Tensor<T> &Left) {
  ArenaArray<T> Data(Left.Data.Size);
  Tensor<T> Result(Left.Dimensions.AllocateCopy(), Data);
  for (size_t i = 0; i < Left.Data.Size; i++) {
    Result[i] = -Left[i];
  }
  return Result;
}

#define TensorInPlaceOperatorOverload(OperatorName, Operator)                  \
  template <typename T> void OperatorName(Tensor<T> &Left, const T Right) {    \
    for (size_t i = 0; i < Left.Data.Size; i++) {                              \
      Left[i] Operator Right;                                                  \
    }                                                                          \
  }                                                                            \
  template <typename T> void OperatorName(Tensor<T> &Left, Tensor<T> &Right) { \
    assert(Left.DimensionsMatch(Right));                                       \
    assert(Left.Data.Size == Right.Data.Size);                                 \
    for (size_t i = 0; i < Left.Data.Size; i++) {                              \
      Left[i] Operator Right[i];                                               \
    }                                                                          \
  }

TensorInPlaceOperatorOverload(operator+=, +=);
TensorInPlaceOperatorOverload(operator-=, -=);
TensorInPlaceOperatorOverload(operator*=, *=);
TensorInPlaceOperatorOverload(operator/=, /=);

#define TensorFunctionApplication(FunctionName, Function)                      \
  template <typename T> Tensor<T> FunctionName(const Tensor<T> &Input) {       \
    ArenaArray<T> Data(Input.Data.Size);                                       \
    Tensor<T> Result(Input.Dimensions.AllocateCopy(), Data);                   \
    for (size_t i = 0; i < Input.Data.Size; i++) {                             \
      Result[i] = Function(Input[i]);                                          \
    }                                                                          \
    return Result;                                                             \
  }                                                                            \
  template <typename T> void FunctionName##InPlace(Tensor<T> &Input) {         \
    for (size_t i = 0; i < Input.Data.Size; i++) {                             \
      Input[i] = Function(Input[i]);                                           \
    }                                                                          \
  }

TensorFunctionApplication(Sqrt, std::sqrt);
TensorFunctionApplication(Exp, std::exp);
TensorFunctionApplication(Log, std::log);
TensorFunctionApplication(Sin, std::sin);
TensorFunctionApplication(Cos, std::cos);
TensorFunctionApplication(Abs, std::abs);

template <typename T> static T ComputeReciprocal(T Number) {
  return static_cast<T>(1) / Number;
}

TensorFunctionApplication(Reciprocal, ComputeReciprocal);

template <typename T> Tensor<T> Pow(const Tensor<T> &Left, const T Right) {
  ArenaArray<T> Data(Left.Data.Size);
  Tensor<T> Result(Left.Dimensions.AllocateCopy(), Data);
  for (size_t i = 0; i < Left.Data.Size; i++) {
    Result[i] = std::pow(Left[i], Right);
  }

  return Result;
}

template <typename T>
Tensor<T> Pow(const Tensor<T> &Left, const Tensor<T> &Right) {
  ArenaArray<T> Data(Left.Data.Size);
  Tensor<T> Result(Left.Dimensions.AllocateCopy(), Data);

  assert(Left.DimensionsMatch(Right));
  assert(Left.Data.Size == Right.Data.Size);

  for (size_t i = 0; i < Left.Data.Size; i++) {
    Result[i] = std::pow(Left[i], Right[i]);
  }

  return Result;
}

template <typename T> void PowInPlace(Tensor<T> &Input, const T Exponent) {
  for (size_t i = 0; i < Input.Data.Size; i++) {
    Input[i] = std::pow(Input[i], Exponent);
  }
}

template <typename T>
void PowInPlace(Tensor<T> &Input, const Tensor<T> &Exponent) {
  assert(Input.DimensionsMatch(Exponent));
  assert(Input.Data.Size == Exponent.Data.Size);

  for (size_t i = 0; i < Input.Data.Size; i++) {
    Input[i] = std::pow(Input[i], Exponent[i]);
  }
}

#endif