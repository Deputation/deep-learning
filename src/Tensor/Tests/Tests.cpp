#include "Tensor.h"
#include "TensorKernels.h"
#include "TensorOperations.h"

#include <Arena.h>
#include <ArenaArray.h>
#include <ArenaWatcher.h>

#include <iostream>

template <typename T>
void MatchesTwoDimensional(const Tensor<T> &Tensor,
                           [[maybe_unused]] const ArenaArray<T> &Array) {
  assert(Tensor.Type == TensorType::Vector ||
         Tensor.Type == TensorType::Matrix ||
         Tensor.Type == TensorType::Scalar);
  assert(Tensor.Data.Size == Array.Size);

  [[maybe_unused]] size_t k = 0;
  for (size_t i = 0; i < Tensor.Dimensions[0]; i++) {
    for (size_t j = 0; j < Tensor.Dimensions[1]; j++) {
      assert((Array[k] == Tensor[i, j]));
      k++;
    }
  }
}

template <typename T>
void MatchesThreeDimensional(const Tensor<T> &Tensor,
                             [[maybe_unused]] const ArenaArray<T> &Array) {
  assert(Tensor.Type == TensorType::Tensor);
  assert(Tensor.Data.Size == Array.Size);

  [[maybe_unused]] size_t a = 0;
  // k must increase last as it selects the sub matrix
  for (size_t k = 0; k < Tensor.Dimensions[2]; k++) {
    for (size_t i = 0; i < Tensor.Dimensions[0]; i++) {
      for (size_t j = 0; j < Tensor.Dimensions[1]; j++) {
        assert((Array[a] == Tensor[{i, j, k}]));
        a++;
      }
    }
  }
}

void TestVectorLayout() {
  ArenaArray<int> TestVectorData = {1, 2, 3};
  Tensor<int> TestVector({3, 1}, {1, 2, 3});
  assert(TestVector.Strides[0] == 1);
  assert(TestVector.Strides[1] == 1);
  assert(TestVector.Type == TensorType::Vector);
  MatchesTwoDimensional(TestVector, TestVectorData);
}

void TestMatrixLayout() {
  ArenaArray<int> TestData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor<int> TestMatrix({4, 3}, {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12});
  assert(TestMatrix.Strides[0] == 1);
  assert(TestMatrix.Strides[1] == 4);
  assert(TestMatrix.Type == TensorType::Matrix);
  MatchesTwoDimensional(TestMatrix, TestData);
}

void TestTensorLayout() {
  ArenaArray<int> TestData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  Tensor<int> TestTensor({4, 3, 3},
                         {1,  4,  7,  10, 2,  5,  8,  11, 3,  6,  9,  12,
                          13, 16, 19, 22, 14, 17, 20, 23, 15, 18, 21, 24,
                          25, 28, 31, 34, 26, 29, 32, 35, 27, 30, 33, 36});

  assert(TestTensor.Strides[0] == 1);
  assert(TestTensor.Strides[1] == 4);
  assert(TestTensor.Strides[2] == 12);
  assert(TestTensor.Type == TensorType::Tensor);
  MatchesThreeDimensional(TestTensor, TestData);
}

void TestGetColumn() {
  ArenaArray<int> FirstTwoColumnsTestData = {1, 2, 4, 5, 7, 8};
  ArenaArray<int> FirstColumnTestData = {1, 4, 7};
  ArenaArray<int> MiddleColumnTestData = {2, 5, 8};
  Tensor<int> TestMatrix({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  auto MiddleColumn = GetColumns(TestMatrix, 1, 1);
  assert(MiddleColumn.Type == TensorType::Vector);
  assert(MiddleColumn.Strides[0] == 1);
  assert(MiddleColumn.Strides[1] == 1);
  MatchesTwoDimensional(MiddleColumn, MiddleColumnTestData);

  auto FastMiddleColumn = FastGetColumns(TestMatrix, 1, 1);
  assert(FastMiddleColumn.Type == TensorType::Vector);
  assert(FastMiddleColumn.Strides[0] == 1);
  assert(FastMiddleColumn.Strides[1] == 1);
  MatchesTwoDimensional(FastMiddleColumn, MiddleColumnTestData);

  auto FirstTwoColumns = GetColumns(TestMatrix, 0, 1);
  assert(FirstTwoColumns.Type == TensorType::Matrix);
  assert(FirstTwoColumns.Strides[0] == 1);
  assert(FirstTwoColumns.Strides[1] == 3);
  MatchesTwoDimensional(FirstTwoColumns, FirstTwoColumnsTestData);

  auto FirstTwoColumnsFast = FastGetColumns(TestMatrix, 0, 1);
  assert(FirstTwoColumnsFast.Type == TensorType::Matrix);
  assert(FirstTwoColumnsFast.Strides[0] == 1);
  assert(FirstTwoColumnsFast.Strides[1] == 3);
  MatchesTwoDimensional(FirstTwoColumnsFast, FirstTwoColumnsTestData);

  auto FirstColumn = GetColumns(TestMatrix, 0, 0);
  assert(FirstColumn.Type == TensorType::Vector);
  assert(FirstColumn.Strides[0] == 1);
  assert(FirstColumn.Strides[1] == 1);
  MatchesTwoDimensional(FirstColumn, FirstColumnTestData);

  auto FirstColumnFast = FastGetColumns(TestMatrix, 0, 0);
  assert(FirstColumnFast.Type == TensorType::Vector);
  assert(FirstColumnFast.Strides[0] == 1);
  assert(FirstColumnFast.Strides[1] == 1);
  MatchesTwoDimensional(FirstColumnFast, FirstColumnTestData);
}

void TestLayout() {
  ArenaWatcher Watcher(GeneralArena);

  TestVectorLayout();
  TestMatrixLayout();
  TestTensorLayout();
  TestGetColumn();
}

void TestMatrixMatrixMultiplication() {
  ArenaArray<int> TestData = {30, 36, 42, 66, 81, 96, 102, 126, 150};

  Tensor<int> FirstTestMatrix({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
  Tensor<int> SecondTestMatrix({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});

  auto Result = MatrixMultiplication(FirstTestMatrix, SecondTestMatrix);
  assert(Result.Dimensions[0] == FirstTestMatrix.Dimensions[0]);
  assert(Result.Dimensions[1] == SecondTestMatrix.Dimensions[1]);
  assert(Result.Type == TensorType::Matrix);
  MatchesTwoDimensional(Result, TestData);
}

void TestMatrixVectorMultiplication() {
  ArenaArray<int> TestData = {14, 32, 50};

  Tensor<int> TestMatrix({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
  Tensor<int> TestVector({3, 1}, {1, 2, 3});

  auto Result = MatrixMultiplication(TestMatrix, TestVector);
  assert(Result.Dimensions[0] == TestMatrix.Dimensions[0]);
  assert(Result.Dimensions[1] == TestVector.Dimensions[1]);
  assert(Result.Type == TensorType::Vector);
  MatchesTwoDimensional(Result, TestData);
}

void TestVectorVectorMultiplication() {
  ArenaArray<int> TestData = {14};

  Tensor<int> FirstTestVector({1, 3}, {1, 2, 3});
  Tensor<int> SecondTestVector({3, 1}, {1, 2, 3});

  auto Result = DotProduct(FirstTestVector, SecondTestVector);

  assert(Result.Dimensions[0] == FirstTestVector.Dimensions[0]);
  assert(Result.Dimensions[1] == SecondTestVector.Dimensions[1]);
  assert(Result.Type == TensorType::Scalar);
  MatchesTwoDimensional(Result, TestData);
}

void TestAddition() {
  ArenaArray<int> TestData = {2, 4, 6};

  Tensor<int> FirstTestVector({1, 3}, {1, 2, 3});
  Tensor<int> SecondTestVector({1, 3}, {1, 2, 3});
  auto Result = FirstTestVector + SecondTestVector;
  MatchesTwoDimensional(Result, TestData);
}

void TestSubtraction() {
  ArenaArray<int> TestData = {0, 0, 0};

  Tensor<int> FirstTestVector({1, 3}, {1, 2, 3});
  Tensor<int> SecondTestVector({1, 3}, {1, 2, 3});
  auto Result = FirstTestVector - SecondTestVector;
  MatchesTwoDimensional(Result, TestData);
}

void TestMultiplication() {
  ArenaArray<int> TestData = {1, 4, 9};

  Tensor<int> FirstTestVector({1, 3}, {1, 2, 3});
  Tensor<int> SecondTestVector({1, 3}, {1, 2, 3});
  auto Result = FirstTestVector * SecondTestVector;
  MatchesTwoDimensional(Result, TestData);
}

void TestDivision() {
  ArenaArray<int> TestData = {1, 1, 1};

  Tensor<int> FirstTestVector({1, 3}, {1, 2, 3});
  Tensor<int> SecondTestVector({1, 3}, {1, 2, 3});
  auto Result = FirstTestVector / SecondTestVector;
  MatchesTwoDimensional(Result, TestData);
}

void TestPower() {
  ArenaArray<int> TestData = {1, 4, 9};

  Tensor<int> FirstTestVector({1, 3}, {1, 2, 3});
  auto Result = Pow(FirstTestVector, 2);
  MatchesTwoDimensional(Result, TestData);
  PowInPlace(FirstTestVector, 2);
  MatchesTwoDimensional(FirstTestVector, TestData);
}

void TestExponential() {
  ArenaArray<int> TestData = {1, 1, 1};

  Tensor<int> FirstTestVector({1, 3}, {0, 0, 0});
  auto Result = Exp(FirstTestVector);
  MatchesTwoDimensional(Result, TestData);
}

void TestElementWiseOperations() {
  TestAddition();
  TestSubtraction();
  TestMultiplication();
  TestDivision();
  TestPower();
  TestExponential();
}

void TestMatrixTransposition() {
  ArenaArray<int> TestData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  ArenaArray<int> TestDataTransposed = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  const Tensor<int> TestMatrix({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
  auto TransposedTestMatrix = Transpose(TestMatrix);

  assert(TestMatrix.Dimensions[0] == 3);
  assert(TestMatrix.Dimensions[1] == 3);
  assert(TestMatrix.Strides[0] == 1);
  assert(TestMatrix.Strides[1] == 3);
  MatchesTwoDimensional(TestMatrix, TestData);

  assert(TransposedTestMatrix.Dimensions[0] == 3);
  assert(TransposedTestMatrix.Dimensions[1] == 3);
  assert(TransposedTestMatrix.Strides[0] == 1);
  assert(TransposedTestMatrix.Strides[1] == 3);
  MatchesTwoDimensional(TransposedTestMatrix, TestDataTransposed);
}

void TestVectorTransposition() {
  ArenaArray<int> TestData = {1, 2, 3};

  Tensor<int> TestVector({1, 3}, {1, 2, 3});
  TransposeInPlace(TestVector);
  assert(TestVector.Dimensions[0] == 3);
  assert(TestVector.Dimensions[1] == 1);
  assert(TestVector.Strides[0] == 1);
  assert(TestVector.Strides[1] == 1);
  MatchesTwoDimensional(TestVector, TestData);

  const Tensor<int> ConstVector({1, 3}, {1, 2, 3});
  auto ConstVectorTransposed = Transpose(ConstVector);
  assert(ConstVectorTransposed.Dimensions[0] == 3);
  assert(ConstVectorTransposed.Dimensions[1] == 1);
  assert(ConstVectorTransposed.Strides[0] == 1);
  assert(ConstVectorTransposed.Strides[1] == 1);
  MatchesTwoDimensional(ConstVectorTransposed, TestData);
}

void TestScalarTransposition() {
  Tensor<int> Scalar({1, 1}, ArenaArray<int>{1});
  TransposeInPlace(Scalar);
  assert(Scalar.Dimensions[0] == 1);
  assert(Scalar.Dimensions[1] == 1);
  assert(Scalar.Data.Size == 1);
  assert(Scalar[0] == 1);
  assert(Scalar.Strides[0] == 1);
  assert(Scalar.Strides[1] == 1);

  const Tensor<int> ConstScalar({1, 1}, ArenaArray<int>{1});
  [[maybe_unused]] auto ConstScalarTransposed = Transpose(ConstScalar);
  assert(ConstScalarTransposed.Dimensions[0] == 1);
  assert(ConstScalarTransposed.Dimensions[1] == 1);
  assert(ConstScalarTransposed.Data.Size == 1);
  assert(ConstScalarTransposed[0] == 1);
  assert(ConstScalarTransposed.Strides[0] == 1);
  assert(ConstScalarTransposed.Strides[1] == 1);
}

void TestTransposition() {
  TestMatrixTransposition();
  TestVectorTransposition();
  TestScalarTransposition();
}

void TestExtension() {
  ArenaArray<int> TestDataRowExtension = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  Tensor<int> RowVector({1, 3}, {1, 2, 3});
  auto RowExtension = ExtendVector(RowVector, 0, 3);
  assert(RowExtension.Dimensions[0] == 3);
  assert(RowExtension.Dimensions[1] == 3);
  MatchesTwoDimensional(RowExtension, TestDataRowExtension);

  ArenaArray<int> TestDataColumnExtension = {1, 1, 1, 2, 2, 2, 3, 3, 3};
  Tensor<int> ColumnVector({3, 1}, {1, 2, 3});
  auto ColumnExtension = ExtendVector(ColumnVector, 1, 3);
  assert(ColumnExtension.Dimensions[0] == 3);
  assert(ColumnExtension.Dimensions[1] == 3);
  MatchesTwoDimensional(ColumnExtension, TestDataColumnExtension);
}

void TestSummation() {
  ArenaArray<int> TestDataRowSum = {6, 15, 24};
  Tensor<int> TestMatrix({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
  auto ResultRowSum = SumMatrix(TestMatrix, 0);
  assert(ResultRowSum.Dimensions[0] == 1);
  assert(ResultRowSum.Dimensions[1] == 3);
  MatchesTwoDimensional(ResultRowSum, TestDataRowSum);

  ArenaArray<int> TestDataColumnSum = {12, 15, 18};
  auto ResultColumnSum = SumMatrix(TestMatrix, 1);
  assert(ResultColumnSum.Dimensions[0] == 3);
  assert(ResultColumnSum.Dimensions[1] == 1);
  MatchesTwoDimensional(ResultColumnSum, TestDataColumnSum);
}

void TestOperations() {
  ArenaWatcher Watcher(GeneralArena);

  TestMatrixMatrixMultiplication();
  TestMatrixVectorMultiplication();
  TestVectorVectorMultiplication();
  TestElementWiseOperations();
  TestTransposition();
  TestExtension();
}

int main([[maybe_unused]] int Argc, [[maybe_unused]] const char *Argv[]) {
  TestLayout();
  TestOperations();

  std::cout << "Tests passed." << std::endl;

  return 0;
}