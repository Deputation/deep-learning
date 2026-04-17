#include "ArenaTensorNode.h"
#include "TensorNodes.h"
#include <Tensor.h>
#include <TensorUtilities.h>

void TestMatrixMatrix() {
  auto Matrix =
      ArenaTensorNode<int>(Tensor<int>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9}));
  auto Matrix2 =
      ArenaTensorNode<int>(Tensor<int>({3, 3}, {1, 1, 1, 2, 1, 1, 1, 1, 1}));
  Print(Matrix2.NodePointer->Value);

  auto Result = MatrixMultiplication<int>(Matrix, Matrix2);
  Print(Result.NodePointer->Value);
  Result = Sum(Result);

  auto Gradient = Tensor<int>(1);
  Result.Backward(Gradient);

  Print(Matrix.GetTensorValuePointer()->Gradient);
  Print(Matrix2.GetTensorValuePointer()->Gradient);
}

void TestVectorVector() {
  auto Matrix =
      ArenaTensorNode<int>(Tensor<int>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9}));
  auto Vector = ArenaTensorNode<int>(Tensor<int>({3, 1}, {1, 2, 1}));

  auto Result = MatrixMultiplication<int>(Matrix, Vector);
  Print(Result.NodePointer->Value);
  Result = Sum(Result);

  auto Gradient = Tensor<int>(1);
  Result.Backward(Gradient);

  Print(Matrix.GetTensorValuePointer()->Gradient);
  Print(Vector.GetTensorValuePointer()->Gradient);
}

int main([[maybe_unused]] int Argc, [[maybe_unused]] const char *Argv[]) {
  TestVectorVector();

  return 0;
}