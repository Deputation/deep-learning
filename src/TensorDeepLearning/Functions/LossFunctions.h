#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "ActivationFunctions.h"
#include "TensorOperations.h"
#include <ArenaTensorNode.h>

template <typename T>
static ArenaTensorNode<T> SquaredError(ArenaTensorNode<T> &Prediction,
                                       Tensor<T> &Label) {
  auto Subtraction = Prediction - Label;

  auto Loss = Pow(Subtraction, static_cast<T>(2));
  Loss /= static_cast<T>(2);

  return Loss;
}

template <typename T>
static ArenaTensorNode<T> CrossEntropy(ArenaTensorNode<T> &Prediction,
                                       Tensor<T> &Label) {
  auto SoftmaxOutput = Softmax(Prediction);
  auto LogOutput = Log(SoftmaxOutput);
  auto Multiplication = LogOutput * Label;
  auto SumResult = SumMatrix(Multiplication, 0);

  return -SumResult;
}

#endif