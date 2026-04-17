#ifndef ARENA_TENSOR_NODE_H
#define ARENA_TENSOR_NODE_H

#include "Arena.h"
#include "TensorNodes.h"

#include <Tensor.h>

template <typename T> class ArenaTensorNode {
public:
  using ArenaTensorNodeType = T;

  TensorNode<T> *NodePointer;

  explicit ArenaTensorNode(Tensor<T> Data) {
    NodePointer = new (GeneralArena.Allocate(sizeof(TensorValue<T>)))
        TensorValue<T>(Data);
  }
  ArenaTensorNode(TensorNode<T> *Node) : NodePointer(Node) {}
  ArenaTensorNode() : NodePointer(nullptr) {}

  TensorValue<T> *GetTensorValuePointer() {
    return (TensorValue<T> *)NodePointer;
  }

  Tensor<T> &GetValue() { return NodePointer->Value; }

  void Backward(Tensor<T> &Gradient) { NodePointer->Backward(Gradient); }

  inline T &operator[](std::initializer_list<size_t> Indices) {
    return GetValue()[Indices];
  }

  inline T &operator[](size_t Row, size_t Column) {
    return GetValue()[Row, Column];
  }

  inline T &operator[](size_t Index) { return GetValue()[Index]; }
};

template <typename T> struct IsArenaTensorNode : std::false_type {};

template <typename T>
struct IsArenaTensorNode<ArenaTensorNode<T>> : std::true_type {};

#define BindUnaryOperatorToNode(OperatorName, NodeClass)                       \
  template <typename T>                                                        \
  static ArenaTensorNode<T> OperatorName(ArenaTensorNode<T> &TermOne) {        \
    return ArenaTensorNode(new (GeneralArena.Allocate(sizeof(NodeClass<T>)))   \
                               NodeClass<T>(TermOne.NodePointer));             \
  }

BindUnaryOperatorToNode(operator-, TensorUnarySubtraction);
BindUnaryOperatorToNode(Sum, SumOperation);
BindUnaryOperatorToNode(Abs, TensorAbsoluteValue);
BindUnaryOperatorToNode(Log, TensorLog);
BindUnaryOperatorToNode(Exp, TensorExp);
BindUnaryOperatorToNode(Sin, TensorSin);

#define BindBinaryOperatorToNode(OperatorName, NodeClass)                      \
  template <typename T>                                                        \
  ArenaTensorNode<T> static OperatorName(ArenaTensorNode<T> &TermOne,          \
                                         ArenaTensorNode<T> &TermTwo) {        \
    return ArenaTensorNode(                                                    \
        new (GeneralArena.Allocate(sizeof(NodeClass<T>)))                      \
            NodeClass<T>(TermOne.NodePointer, TermTwo.NodePointer));           \
  }

BindBinaryOperatorToNode(operator+, TensorAddition);
BindBinaryOperatorToNode(operator-, TensorSubtraction);
BindBinaryOperatorToNode(operator*, TensorMultiplication);
BindBinaryOperatorToNode(operator/, TensorDivision);

#define BindBinaryOperatorToConstantNode(OperatorName, NodeClass)              \
  template <typename T>                                                        \
  ArenaTensorNode<T> static OperatorName(ArenaTensorNode<T> &TermOne,          \
                                         T Constant) {                         \
    return ArenaTensorNode(new (GeneralArena.Allocate(sizeof(NodeClass<T>)))   \
                               NodeClass<T>(TermOne.NodePointer, Constant));   \
  }

BindBinaryOperatorToConstantNode(operator+, TensorConstantAddition);
BindBinaryOperatorToConstantNode(operator-, TensorConstantSubtraction);
BindBinaryOperatorToConstantNode(operator*, TensorConstantMultiplication);
BindBinaryOperatorToConstantNode(operator/, TensorConstantDivision);

#define BindBinaryOperatorToConstantTensorNode(OperatorName, NodeClass)        \
  template <typename T>                                                        \
  ArenaTensorNode<T> static OperatorName(ArenaTensorNode<T> &TermOne,          \
                                         Tensor<T> Constant) {                 \
    return ArenaTensorNode(new (GeneralArena.Allocate(sizeof(NodeClass<T>)))   \
                               NodeClass<T>(TermOne.NodePointer, Constant));   \
  }

BindBinaryOperatorToConstantTensorNode(operator+, TensorConstantTensorAddition);
BindBinaryOperatorToConstantTensorNode(operator-,
                                       TensorConstantTensorSubtraction);
BindBinaryOperatorToConstantTensorNode(operator*,
                                       TensorConstantTensorMultiplication);
BindBinaryOperatorToConstantTensorNode(operator/, TensorConstantTensorDivision);

#define BindInPlaceBinaryOperator(OperatorName, Operator)                      \
  template <typename T>                                                        \
  static void OperatorName(ArenaTensorNode<T> &TermOne,                        \
                           ArenaTensorNode<T> &TermTwo) {                      \
    auto Result = TermOne Operator TermTwo;                                    \
    TermOne = Result;                                                          \
  }                                                                            \
  template <typename T>                                                        \
  static void OperatorName(ArenaTensorNode<T> &TermOne, T TermTwo) {           \
    auto Result = TermOne Operator TermTwo;                                    \
    TermOne = Result;                                                          \
  }                                                                            \
  template <typename T>                                                        \
  static void OperatorName(ArenaTensorNode<T> &TermOne, Tensor<T> TermTwo) {   \
    auto Result = TermOne Operator TermTwo;                                    \
    TermOne = Result;                                                          \
  }

BindInPlaceBinaryOperator(operator+=, +);
BindInPlaceBinaryOperator(operator-=, -);
BindInPlaceBinaryOperator(operator*=, *);
BindInPlaceBinaryOperator(operator/=, /);

BindBinaryOperatorToNode(Pow, TensorPower);
BindBinaryOperatorToConstantNode(Pow, TensorConstantPower);

BindBinaryOperatorToNode(DotProduct, DotProductOperation);
BindBinaryOperatorToNode(MatrixVectorMultiplication,
                         MatrixVectorMultiplicationOperation);
BindBinaryOperatorToNode(MatrixMatrixMultiplication,
                         MatrixMatrixMultiplicationOperation);

BindBinaryOperatorToConstantTensorNode(DotProduct, DotProductConstantOperation);
BindBinaryOperatorToConstantTensorNode(
    MatrixVectorMultiplication, MatrixConstantVectorMultiplicationOperation);
BindBinaryOperatorToConstantTensorNode(
    MatrixMatrixMultiplication, MatrixConstantMatrixMultiplicationOperation);

template <typename T>
static ArenaTensorNode<T> MatrixMultiplication(ArenaTensorNode<T> &Left,
                                               ArenaTensorNode<T> &Right) {
  assert(Left.NodePointer->Value.Type != TensorType::Tensor);
  assert(Right.NodePointer->Value.Type != TensorType::Tensor);

  if (Left.NodePointer->Value.Type == TensorType::Vector &&
      Right.NodePointer->Value.Type == TensorType::Vector) {
    return DotProduct(Left, Right);
  }

  if (Left.NodePointer->Value.Type == TensorType::Matrix &&
      Right.NodePointer->Value.Type == TensorType::Vector) {
    return MatrixVectorMultiplication(Left, Right);
  }

  return MatrixMatrixMultiplication(Left, Right);
}

template <typename T>
static ArenaTensorNode<T> MatrixMultiplication(ArenaTensorNode<T> &Left,
                                               Tensor<T> Right) {
  assert(Left.NodePointer->Value.Type != TensorType::Tensor);
  assert(Right.Type != TensorType::Tensor);

  if (Left.NodePointer->Value.Type == TensorType::Vector &&
      Right.Type == TensorType::Vector) {
    return DotProduct(Left, Right);
  }

  if (Left.NodePointer->Value.Type == TensorType::Matrix &&
      Right.Type == TensorType::Vector) {
    return MatrixVectorMultiplication(Left, Right);
  }

  return MatrixMatrixMultiplication(Left, Right);
}

template <typename T>
static ArenaTensorNode<T> SumMatrix(ArenaTensorNode<T> &Input, size_t Index) {
  return new (GeneralArena.Allocate(sizeof(SumMatrixOperation<T>)))
      SumMatrixOperation(Input.NodePointer, Index);
}

template <typename T>
static ArenaTensorNode<T> ExtendVector(ArenaTensorNode<T> &Input, size_t Index,
                                       size_t NewSize) {
  return new (GeneralArena.Allocate(sizeof(ExtendVectorOperation<T>)))
      ExtendVectorOperation(Input.NodePointer, Index, NewSize);
}

template <typename T>
static ArenaTensorNode<T> ExtendToTensor(ArenaTensorNode<T> &Input,
                                         ArenaArray<size_t> Dimensions) {
  return new (GeneralArena.Allocate(sizeof(ExtendToTensorOperation<T>)))
      ExtendToTensorOperation(Input.NodePointer, Dimensions);
}

template <typename T>
static ArenaTensorNode<T> Average(ArenaTensorNode<T> &Input) {
  assert(Input.NodePointer->Value.Type == TensorType::Vector);

  auto Average = Sum(Input);
  Average /= static_cast<T>(Input.NodePointer->Value.Data.Size);

  return Average;
}

#endif