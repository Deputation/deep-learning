#ifndef BACKWARD_MODE_H
#define BACKWARD_MODE_H

#include <Tensor.h>
#include <TensorKernels.h>
#include <TensorOperations.h>

#include <Arena.h>

template <typename T> class TensorNode {
public:
  Tensor<T> Value;

  virtual void Backward(Tensor<T> &UpstreamGradient) = 0;
};

template <typename T> class TensorValue : public TensorNode<T> {
public:
  Tensor<T> Gradient;

  TensorValue(Tensor<T> Value) {
    this->Value = Value;
    Gradient = Tensor<T>(Value.Dimensions.AllocateCopy());
  }

  void Backward(Tensor<T> &UpstreamGradient) { Gradient += UpstreamGradient; }
  void ZeroGradient() {
    for (size_t i = 0; i < Gradient.Data.Size; i++) {
      Gradient[i] = static_cast<T>(0);
    }
  }
};

template <typename T> class TensorUnarySubtraction : public TensorNode<T> {
public:
  TensorNode<T> *Term;

  TensorUnarySubtraction(TensorNode<T> *Term) : Term(Term) {
    this->Value = -Term->Value;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto Gradient = -UpstreamGradient;
    Term->Backward(Gradient);
  }
};

template <typename T> class TensorAddition : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  TensorAddition(TensorNode<T> *TermOne, TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TermTwo->Value.Type);

    this->Value = TermOne->Value + TermTwo->Value;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    TermOne->Backward(UpstreamGradient);
    TermTwo->Backward(UpstreamGradient);
  }
};

template <typename T> class TensorConstantAddition : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  T Constant;

  TensorConstantAddition(TensorNode<T> *TermOne, T Constant)
      : TermOne(TermOne), Constant(Constant) {
    this->Value = TermOne->Value + Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    TermOne->Backward(UpstreamGradient);
  }
};

template <typename T>
class TensorConstantTensorAddition : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  TensorConstantTensorAddition(TensorNode<T> *TermOne, Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == Constant.Type);

    this->Value = TermOne->Value + Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    TermOne->Backward(UpstreamGradient);
  }
};

template <typename T> class TensorSubtraction : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  TensorSubtraction(TensorNode<T> *TermOne, TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TermTwo->Value.Type);

    this->Value = TermOne->Value - TermTwo->Value;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    TermOne->Backward(UpstreamGradient);

    auto Gradient = -UpstreamGradient;
    TermTwo->Backward(Gradient);
  }
};

template <typename T> class TensorConstantSubtraction : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  T Constant;

  TensorConstantSubtraction(TensorNode<T> *TermOne, T Constant)
      : TermOne(TermOne), Constant(Constant) {
    this->Value = TermOne->Value - Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    TermOne->Backward(UpstreamGradient);
  }
};

template <typename T>
class TensorConstantTensorSubtraction : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  TensorConstantTensorSubtraction(TensorNode<T> *TermOne, Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == Constant.Type);

    this->Value = TermOne->Value - Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    TermOne->Backward(UpstreamGradient);
  }
};

template <typename T> class TensorMultiplication : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  TensorMultiplication(TensorNode<T> *TermOne, TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TermTwo->Value.Type);

    this->Value = TermOne->Value * TermTwo->Value;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto TermOneGradient = TermTwo->Value * UpstreamGradient;
    TermOne->Backward(TermOneGradient);

    auto TermTwoGradient = TermOne->Value * UpstreamGradient;
    TermTwo->Backward(TermTwoGradient);
  }
};

template <typename T>
class TensorConstantMultiplication : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  T Constant;

  TensorConstantMultiplication(TensorNode<T> *TermOne, T Constant)
      : TermOne(TermOne), Constant(Constant) {
    this->Value = TermOne->Value * Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto TermOneGradient = UpstreamGradient * Constant;
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T>
class TensorConstantTensorMultiplication : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  TensorConstantTensorMultiplication(TensorNode<T> *TermOne, Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == Constant.Type);

    this->Value = TermOne->Value * Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto TermOneGradient = UpstreamGradient * Constant;
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T> class TensorDivision : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  TensorDivision(TensorNode<T> *TermOne, TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TermTwo->Value.Type);

    this->Value = TermOne->Value / TermTwo->Value;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto ArenaNumber = Tensor<T>(TermTwo->Value.Dimensions.AllocateCopy(), 2);

    auto TermOneGradient =
        (TermTwo->Value / Pow(TermTwo->Value, ArenaNumber)) * UpstreamGradient;
    TermOne->Backward(TermOneGradient);
    auto TermTwoGradient =
        (-TermOne->Value / Pow(TermTwo->Value, ArenaNumber)) * UpstreamGradient;
    TermTwo->Backward(TermTwoGradient);
  }
};

template <typename T> class TensorConstantDivision : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  T Constant;

  TensorConstantDivision(TensorNode<T> *TermOne, T Constant)
      : TermOne(TermOne), Constant(Constant) {
    this->Value = TermOne->Value / Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto PowerResult = std::pow(Constant, 2);

    auto TermOneGradient = UpstreamGradient * (Constant / PowerResult);
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T>
class TensorConstantTensorDivision : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  TensorConstantTensorDivision(TensorNode<T> *TermOne, Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == Constant.Type);

    this->Value = TermOne->Value / Constant;
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto PowerResult = Pow(Constant, 2);

    auto TermOneGradient = UpstreamGradient * (Constant / PowerResult);
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T> class TensorPower : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  TensorPower(TensorNode<T> *TermOne, TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TermTwo->Value.Type);

    this->Value = Pow(TermOne->Value, TermTwo->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto PowerRule = Pow(TermTwo->Value * TermOne->Value, TermTwo->Value - 1);
    auto ExponentialRule =
        Pow(TermOne->Value, TermTwo->Value) * Log(TermOne->Value);

    auto TermOneGradient = UpstreamGradient * PowerRule;
    TermOne->Backward(TermOneGradient);
    auto TermTwoGradient = UpstreamGradient * ExponentialRule;
    TermTwo->Backward(TermTwoGradient);
  }
};

template <typename T> class TensorConstantPower : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  T Constant;

  TensorConstantPower(TensorNode<T> *TermOne, T Constant)
      : TermOne(TermOne), Constant(Constant) {
    this->Value = Pow(TermOne->Value, Constant);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto MultipliedTermOne = TermOne->Value * Constant;
    auto PowerRule = Pow(MultipliedTermOne, Constant - 1);

    auto TermOneGradient = UpstreamGradient * PowerRule;
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T> class TensorConstantTensorPower : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  TensorConstantTensorPower(TensorNode<T> *TermOne, Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == Constant.Type);

    this->Value = Pow(TermOne->Value, Constant);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto MultipliedTermOne = TermOne->Value * Constant;
    auto SubtractedConstant = Constant - 1;
    auto PowerRule = Pow(MultipliedTermOne, Constant);

    auto TermOneGradient = UpstreamGradient * PowerRule;
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T> class TensorAbsoluteValue : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;

  TensorAbsoluteValue(TensorNode<T> *TermOne) : TermOne(TermOne) {
    this->Value = Abs(TermOne->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto Gradient = Tensor<T>(TermOne->Value.Dimensions.AllocateCopy());
    for (size_t i = 0; i < UpstreamGradient.Data.Size; i++) {
      assert(UpstreamGradient[i] != 0);
      Gradient[i] =
          TermOne->Value[i] > 0 ? UpstreamGradient[i] : -UpstreamGradient[i];
    }
    TermOne->Backward(Gradient);
  }
};

template <typename T> class TensorSin : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;

  TensorSin(TensorNode<T> *TermOne) : TermOne(TermOne) {
    this->Value = Sin(TermOne->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto Gradient = Cos(TermOne->Value) * UpstreamGradient;
    TermOne->Backward(Gradient);
  }
};

template <typename T> class TensorExp : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;

  TensorExp(TensorNode<T> *TermOne) : TermOne(TermOne) {
    this->Value = Exp(TermOne->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto Gradient = Exp(TermOne->Value) * UpstreamGradient;
    TermOne->Backward(Gradient);
  }
};

template <typename T> class TensorLog : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;

  TensorLog(TensorNode<T> *TermOne) : TermOne(TermOne) {
    this->Value = Log(TermOne->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    auto Gradient = Reciprocal(TermOne->Value) * UpstreamGradient;
    TermOne->Backward(Gradient);
  }
};

template <typename T> class DotProductOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  DotProductOperation(TensorNode<T> *TermOne, TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TensorType::Vector);
    assert(TermTwo->Value.Type == TensorType::Vector);

    this->Value = DotProduct(TermOne->Value, TermTwo->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Scalar);

    auto dTermOne = TermTwo->Value * UpstreamGradient;
    // The result must be transposed to match the dimensions of TermOne
    TransposeVectorInPlace(dTermOne);
    TermOne->Backward(dTermOne);

    auto dTermTwo = TermOne->Value * UpstreamGradient;
    // The result must be transposed to match the dimensions of TermTwo
    TransposeVectorInPlace(dTermTwo);
    TermTwo->Backward(dTermTwo);
  }
};

template <typename T> class DotProductConstantOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  DotProductConstantOperation(TensorNode<T> *TermOne, Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == TensorType::Vector);
    assert(Constant.Type == TensorType::Vector);

    this->Value = DotProduct(TermOne->Value, Constant);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Scalar);

    auto dTermOne = Constant * UpstreamGradient;
    // The result must be transposed to match the dimensions of TermOne
    TransposeVectorInPlace(dTermOne);
    TermOne->Backward(dTermOne);
  }
};

template <typename T>
class MatrixVectorMultiplicationOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  MatrixVectorMultiplicationOperation(TensorNode<T> *TermOne,
                                      TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TensorType::Matrix);
    assert(TermTwo->Value.Type == TensorType::Vector);

    this->Value = MatrixMultiplication(TermOne->Value, TermTwo->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Vector);

    auto MatrixGradient = OuterProduct(UpstreamGradient, TermTwo->Value);
    TermOne->Backward(MatrixGradient);

    auto VectorGradient =
        MatrixMultiplication(TermOne->Value, UpstreamGradient, true, false);
    TermTwo->Backward(VectorGradient);
  }
};

template <typename T>
class MatrixConstantVectorMultiplicationOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  MatrixConstantVectorMultiplicationOperation(TensorNode<T> *TermOne,
                                              Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == TensorType::Matrix);
    assert(Constant.Type == TensorType::Vector);

    this->Value = MatrixMultiplication(TermOne->Value, Constant);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Vector);

    auto MatrixGradient = OuterProduct(UpstreamGradient, Constant);
    TermOne->Backward(MatrixGradient);
  }
};

template <typename T>
class MatrixMatrixMultiplicationOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  TensorNode<T> *TermTwo;

  MatrixMatrixMultiplicationOperation(TensorNode<T> *TermOne,
                                      TensorNode<T> *TermTwo)
      : TermOne(TermOne), TermTwo(TermTwo) {
    assert(TermOne->Value.Type == TensorType::Matrix);
    assert(TermTwo->Value.Type == TensorType::Matrix);

    this->Value = MatrixMultiplication(TermOne->Value, TermTwo->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Matrix);

    auto TermOneGradient =
        MatrixMultiplication(UpstreamGradient, TermTwo->Value, false, true);
    TermOne->Backward(TermOneGradient);

    auto TermTwoGradient =
        MatrixMultiplication(TermOne->Value, UpstreamGradient, true, false);
    TermTwo->Backward(TermTwoGradient);
  }
};

template <typename T>
class MatrixConstantMatrixMultiplicationOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  Tensor<T> Constant;

  MatrixConstantMatrixMultiplicationOperation(TensorNode<T> *TermOne,
                                              Tensor<T> Constant)
      : TermOne(TermOne), Constant(Constant) {
    assert(TermOne->Value.Type == TensorType::Matrix);
    assert(Constant.Type == TensorType::Matrix);

    this->Value = MatrixMultiplication(TermOne->Value, Constant);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Matrix);

    auto TermOneGradient =
        MatrixMultiplication(UpstreamGradient, Constant, false, true);
    TermOne->Backward(TermOneGradient);
  }
};

template <typename T> class SumOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;

  SumOperation(TensorNode<T> *TermOne) : TermOne(TermOne) {
    this->Value = Sum(TermOne->Value);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Scalar);
    auto Gradient = ExtendToTensor(UpstreamGradient,
                                   TermOne->Value.Dimensions.AllocateCopy());
    TermOne->Backward(Gradient);
  }
};

template <typename T> class SumMatrixOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  size_t Index;
  size_t ExtensionValue;

  SumMatrixOperation(TensorNode<T> *TermOne, size_t Index)
      : TermOne(TermOne), Index(Index) {
    assert(TermOne->Value.Type == TensorType::Matrix);
    ExtensionValue = TermOne->Value.Dimensions[Index];

    this->Value = SumMatrix(TermOne->Value, Index);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Vector);

    auto Gradient = ExtendVector(UpstreamGradient, Index, ExtensionValue);
    TermOne->Backward(Gradient);
  }
};

template <typename T> class ExtendVectorOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;
  size_t Index;

  ExtendVectorOperation(TensorNode<T> *TermOne, size_t Index, size_t NewSize)
      : TermOne(TermOne), Index(Index) {
    assert(TermOne->Value.Type == TensorType::Vector);

    this->Value = ExtendVector(TermOne->Value, Index, NewSize);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type == TensorType::Matrix);

    auto Gradient = SumMatrix(UpstreamGradient, Index);
    TermOne->Backward(Gradient);
  }
};

template <typename T> class ExtendToTensorOperation : public TensorNode<T> {
public:
  TensorNode<T> *TermOne;

  ExtendToTensorOperation(TensorNode<T> *TermOne, ArenaArray<size_t> Dimensions)
      : TermOne(TermOne) {
    assert(TermOne->Value.Type == TensorType::Scalar);

    this->Value = ExtendToTensor(TermOne->Value, Dimensions);
  }

  void Backward(Tensor<T> &UpstreamGradient) {
    assert(UpstreamGradient.Type != TensorType::Scalar);

    auto Gradient = Sum(UpstreamGradient);
    TermOne->Backward(Gradient);
  }
};

#endif