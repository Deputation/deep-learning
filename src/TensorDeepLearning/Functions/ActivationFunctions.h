#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <ArenaTensorNode.h>
#include <TensorOperations.h>
#include <TensorUtilities.h>

template <typename T>
Tensor<T> SinNoGradient(Tensor<T>& Value) { return Sin(Value); }

template <typename T>
ArenaTensorNode<T> SinActivation(ArenaTensorNode<T> &Value) {
  return Sin(Value);
}

template <typename T> ArenaTensorNode<T> ReLU(ArenaTensorNode<T> &Value) {
  auto AbsValue = Abs(Value);
  auto Factor = Value + AbsValue;
  Factor /= static_cast<T>(2);
  return Factor;
}

template <typename T> Tensor<T> Softmax(Tensor<T> &Value) {
  auto ExponentialSum = Exp(Value);
  if (ExponentialSum.Dimensions[1] > 1) {
    ExponentialSum = SumMatrix(ExponentialSum, 0);
    ExponentialSum = ExtendVector(ExponentialSum, 0, Value.Dimensions[0]);
  } else {
    ExponentialSum = Sum(ExponentialSum);
    ExponentialSum =
        ExtendToTensor(ExponentialSum, Value.Dimensions.AllocateCopy());
  }

  auto ExponentialVector = Exp(Value);
  auto Result = ExponentialVector / ExponentialSum;

  return Result;
}

template <typename T> ArenaTensorNode<T> Softmax(ArenaTensorNode<T> &Value) {
  auto ExponentialSum = Exp(Value);
  if (ExponentialSum.NodePointer->Value.Dimensions[1] > 1) {
    ExponentialSum = SumMatrix(ExponentialSum, 0);
    ExponentialSum =
        ExtendVector(ExponentialSum, 0, Value.NodePointer->Value.Dimensions[0]);
  } else {
    ExponentialSum = Sum(ExponentialSum);
    ExponentialSum = ExtendToTensor(
        ExponentialSum, Value.NodePointer->Value.Dimensions.AllocateCopy());
  }

  auto ExponentialVector = Exp(Value);
  auto Result = ExponentialVector / ExponentialSum;

  return Result;
}

#endif