#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

#include "Tensor.h"

template <typename T> static size_t Max(const Tensor<T> &Input) {
  size_t Result = 0;
  for (size_t i = 1; i < Input.Data.Size; i++) {
    if (Input[Result] < Input[i]) {
      Result = i;
    }
  }
  return Result;
}

template <typename T>
static Tensor<T> MatrixMax(const Tensor<T> &Input, size_t Index) {
  assert(Input.Type == TensorType::Matrix);
  assert(Index == 0 || Index == 1);

  if (Index == 0) {
    Tensor<T> Result({Input.Dimensions[0], 1});

    size_t ResultIndex = 0;
    for (size_t i = 0; i < Input.Dimensions[0]; i++) {
      size_t MaxColIndex = 0;
      for (size_t j = 1; j < Input.Dimensions[1]; j++) {
        if (Input[i, MaxColIndex] < Input[i, j]) {
          MaxColIndex = j;
        }
      }
      Result[ResultIndex, 0] = Input[i, MaxColIndex];
      ResultIndex++;
    }

    return Result;
  }

  if (Index == 1) {
    Tensor<T> Result({1, Input.Dimensions[1]});
    size_t ResultIndex = 0;
    for (size_t i = 0; i < Input.Dimensions[1]; i++) {
      size_t MaxRowIndex = 0;
      for (size_t j = 0; j < Input.Dimensions[0]; j++) {
        if (Input[MaxRowIndex, i] < Input[j, i]) {
          MaxRowIndex = j;
        }
      }
      Result[0, ResultIndex] = Input[MaxRowIndex, i];
      ResultIndex++;
    }
    return Result;
  }

  assert(false);
  return Tensor<T>();
}

template <typename T> static Tensor<T> Sum(const Tensor<T> &Input) {
  T Result = 0;
  for (size_t i = 0; i < Input.Data.Size; i++) {
    Result += Input[i];
  }
  return Result;
}

template <typename T> static Tensor<T> Average(const Tensor<T> &Input) {
  T Result = 0;
  for (size_t i = 0; i < Input.Data.Size; i++) {
    Result += Input[i];
  }
  return Result / Input.Data.Size;
}

template <typename T>
static Tensor<T> ExtendVector(const Tensor<T> &Input, size_t Index,
                              size_t NewSize) {
  assert(Input.Type == TensorType::Vector);
  assert(Index < Input.Dimensions.Size);
  assert(Input.Dimensions[Index] == 1);

  ArenaArray<size_t> NewDimensions(Input.Dimensions.Size);
  for (size_t i = 0; i < NewDimensions.Size; i++) {
    if (i == Index) {
      NewDimensions[i] = NewSize;
      continue;
    }
    NewDimensions[i] = Input.Dimensions[i];
  }
  Tensor<T> NewTensor(NewDimensions);

  if (Index == 0) {
    for (size_t i = 0; i < NewSize; i++) {
      for (size_t j = 0; j < Input.Dimensions[0]; j++) {
        for (size_t k = 0; k < Input.Dimensions[1]; k++) {
          NewTensor[i, k] = Input[j, k];
        }
      }
    }
  }

  if (Index == 1) {
    for (size_t i = 0; i < NewSize; i++) {
      for (size_t j = 0; j < Input.Dimensions[1]; j++) {
        for (size_t k = 0; k < Input.Dimensions[0]; k++) {
          NewTensor[k, i] = Input[k, j];
        }
      }
    }
  }

  return NewTensor;
}

template <typename T>
static Tensor<T> SumMatrix(const Tensor<T> &Input, size_t Index) {
  assert(Input.Type == TensorType::Matrix);
  assert(Index < Input.Dimensions.Size);
  assert(Input.Dimensions[Index] > 1);

  ArenaArray<size_t> NewDimensions(Input.Dimensions.Size);
  for (size_t i = 0; i < NewDimensions.Size; i++) {
    if (i == Index) {
      NewDimensions[i] = 1;
      continue;
    }
    NewDimensions[i] = Input.Dimensions[i];
  }
  Tensor<T> NewTensor(NewDimensions);

  if (Index == 0) {
    for (size_t i = 0; i < Input.Dimensions[0]; i++) {
      for (size_t j = 0; j < Input.Dimensions[1]; j++) {
        NewTensor[0, j] += Input[i, j];
      }
    }
  }

  if (Index == 1) {
    for (size_t i = 0; i < Input.Dimensions[0]; i++) {
      for (size_t j = 0; j < Input.Dimensions[1]; j++) {
        NewTensor[i, 0] += Input[i, j];
      }
    }
  }

  return NewTensor;
}

template <typename T>
static Tensor<T> ExtendToTensor(const Tensor<T> &Input,
                                ArenaArray<size_t> Dimensions) {
  assert(Input.Type == TensorType::Scalar);

  return Tensor<T>(Dimensions, Input.Data[0]);
}

template <typename T>
static Tensor<T> OuterProduct(const Tensor<T> &Left, const Tensor<T> &Right) {
  assert(Left.Type == TensorType::Vector && Left.Dimensions[1] == 1);
  assert(Right.Type == TensorType::Vector && Right.Dimensions[1] == 1);

  Tensor<T> Result(
      ArenaArray<size_t>({Left.Dimensions[0], Right.Dimensions[0]}));
  for (size_t i = 0; i < Result.Dimensions[0]; i++) {
    for (size_t j = 0; j < Result.Dimensions[1]; j++) {
      for (size_t k = 0; k < Left.Dimensions[0]; k++) {
        Result[k, j] = Left[i, 0] * Right[j, 0];
      }
    }
  }

  return Result;
}

template <typename T>
static Tensor<T> DotProduct(const Tensor<T> &Left, const Tensor<T> &Right) {
  assert(Left.Type == Right.Type);
  assert(Left.Dimensions[1] == Right.Dimensions[0]);
  assert(Left.Data.Size == Right.Data.Size);

  T Sum = 0;
  for (size_t i = 0; i < Left.Data.Size; i++) {
    Sum += Left[i] * Right[i];
  }

  return Tensor<T>({1, 1}, {Sum});
}

template <typename T>
static Tensor<T> MatrixMultiplicationNoTranspose(const Tensor<T> &Left,
                                                 const Tensor<T> &Right) {
  auto RowLength = Left.Dimensions[1];

  Tensor<T> ResultTensor(
      ArenaArray<size_t>({Left.Dimensions[0], Right.Dimensions[1]}));

  for (size_t i = 0; i < Left.Dimensions[0]; i++) {
    for (size_t k = 0; k < RowLength; k++) {
      for (size_t j = 0; j < Right.Dimensions[1]; j++) {
        ResultTensor[i, j] += Left[i, k] * Right[k, j];
      }
    }
  }

  return ResultTensor;
}

template <typename T>
static Tensor<T> MatrixMultiplicationTransposeLeft(const Tensor<T> &Left,
                                                   const Tensor<T> &Right) {
  auto RowLength = Left.Dimensions[0];

  Tensor<T> ResultTensor(
      ArenaArray<size_t>({Left.Dimensions[1], Right.Dimensions[1]}));

  for (size_t i = 0; i < Left.Dimensions[1]; i++) {
    for (size_t k = 0; k < RowLength; k++) {
      for (size_t j = 0; j < Right.Dimensions[1]; j++) {
        ResultTensor[i, j] += Left[k, i] * Right[k, j];
      }
    }
  }

  return ResultTensor;
}

template <typename T>
static Tensor<T> MatrixMultiplicationTransposeRight(const Tensor<T> &Left,
                                                    const Tensor<T> &Right) {
  auto RowLength = Left.Dimensions[1];

  Tensor<T> ResultTensor(
      ArenaArray<size_t>({Left.Dimensions[0], Right.Dimensions[0]}));

  for (size_t i = 0; i < Left.Dimensions[0]; i++) {
    for (size_t k = 0; k < RowLength; k++) {
      for (size_t j = 0; j < Right.Dimensions[0]; j++) {
        ResultTensor[i, j] += Left[i, k] * Right[j, k];
      }
    }
  }

  return ResultTensor;
}

template <typename T>
static Tensor<T>
MatrixMultiplicationTransposeLeftRight(const Tensor<T> &Left,
                                       const Tensor<T> &Right) {
  auto RowLength = Left.Dimensions[0];

  Tensor<T> ResultTensor(
      ArenaArray<size_t>({Left.Dimensions[1], Right.Dimensions[0]}));

  for (size_t i = 0; i < Left.Dimensions[1]; i++) {
    for (size_t k = 0; k < RowLength; k++) {
      for (size_t j = 0; j < Right.Dimensions[0]; j++) {
        ResultTensor[i, j] += Left[k, i] * Right[j, k];
      }
    }
  }

  return ResultTensor;
}

template <typename T>
static Tensor<T>
MatrixMultiplication(const Tensor<T> &Left, const Tensor<T> &Right,
                     bool TransposeLeft = false, bool TransposeRight = false) {
  assert(Left.Type == TensorType::Matrix);
  assert(Right.Type == TensorType::Matrix || Right.Type == TensorType::Vector);

  if (!TransposeLeft && !TransposeRight) {
    assert(Left.Dimensions[1] == Right.Dimensions[0]);
    return MatrixMultiplicationNoTranspose(Left, Right);
  }

  if (TransposeLeft && !TransposeRight) {
    assert(Left.Dimensions[0] == Right.Dimensions[0]);
    return MatrixMultiplicationTransposeLeft(Left, Right);
  }

  if (!TransposeLeft && TransposeRight) {
    assert(Left.Dimensions[1] == Right.Dimensions[1]);
    return MatrixMultiplicationTransposeRight(Left, Right);
  }

  assert(Left.Dimensions[0] == Right.Dimensions[1]);
  return MatrixMultiplicationTransposeLeftRight(Left, Right);
}

template <typename T>
static Tensor<T> PadTensor(Tensor<T> &Input, size_t Padding) {
  assert(Input.IsMatrix());

  auto Dimensions = Input.Dimensions.AllocateCopy();
  Dimensions[0] += Padding * 2;
  Dimensions[1] += Padding * 2;

  ::Tensor<T> NewTensor(Dimensions);

  T PaddingElement;
  PaddingElement = static_cast<T>(0);

  for (size_t i = 0; i < NewTensor.Data.Size; i++) {
    NewTensor.Data[i] = PaddingElement;
  }

  for (size_t i = 0; i < Input.Dimensions[0]; i++) {
    for (size_t j = 0; j < Input.Dimensions[1]; j++) {
      NewTensor.Access(i + Padding, j + Padding) = Input.Access(i, j);
    }
  }

  return NewTensor;
}

template <typename T> static void TransposeVectorInPlace(Tensor<T> &Input) {
  std::swap(Input.Dimensions[0], Input.Dimensions[1]);
}

template <typename T> static void TransposeInPlace(Tensor<T> &Input) {
  switch (Input.Type) {
  case TensorType::Scalar:
    return;
  case TensorType::Vector:
    std::swap(Input.Dimensions[0], Input.Dimensions[1]);
    return;
  case TensorType::Matrix:
  case TensorType::Tensor:
    assert(false);
    return;
  }
}

template <typename T> static Tensor<T> Transpose(const Tensor<T> &Input) {
  switch (Input.Type) {
  case TensorType::Scalar: {
    return Tensor<T>(Input.Dimensions.AllocateCopy(),
                     Input.Data.AllocateCopy());
  }
  case TensorType::Vector: {
    Tensor<T> Result({Input.Dimensions[1], Input.Dimensions[0]},
                     Input.Data.AllocateCopy());
    return Result;
  }
  case TensorType::Matrix:
    assert(Input.Dimensions[0] == Input.Dimensions[1]);
    break;
  case TensorType::Tensor:
    assert(false);
    break;
  }

  ArenaArray<T> Data(Input.Data.Size);
  Tensor<T> Result(Input.Dimensions.AllocateCopy(), Data);

  for (size_t i = 0; i < Input.Dimensions[0]; i++) {
    for (size_t j = 0; j < Input.Dimensions[1]; j++) {
      Result[i, j] = Input[j, i];
    }
  }

  return Result;
}

template <typename T>
static void Copy(Tensor<T> &DestinationTensor, Tensor<T> &SourceTensor) {
  assert(DestinationTensor.Data.Size == SourceTensor.Data.Size);
  assert(DestinationTensor.DimensionsMatch(SourceTensor));

  memcpy(&DestinationTensor.Data[0], &SourceTensor.Data[0],
         SourceTensor.Data.Size * sizeof(T));
}

template <typename T>
static Tensor<T> GetColumns(Tensor<T> &Input, size_t StartIndex,
                            size_t EndIndex) {
  [[maybe_unused]] auto Difference =
      static_cast<long>(EndIndex) - static_cast<long>(StartIndex);
  assert(Difference >= 0);
  auto ColumnsFromStart = EndIndex - StartIndex;
  assert(Input.Type == TensorType::Matrix);
  assert(EndIndex < Input.Dimensions[1]);
  auto ColumnLength = Input.Strides[1];

  Tensor<T> NewTensor(
      ArenaArray<size_t>({Input.Dimensions[0], ColumnsFromStart + 1}));
  memcpy(&NewTensor[0], &Input[StartIndex * Input.Strides[1]],
         ((ColumnsFromStart + 1) * ColumnLength) * sizeof(T));
  assert(((ColumnsFromStart + 1) * ColumnLength) == NewTensor.Data.Size);

  return NewTensor;
}

template <typename T>
static Tensor<T> FastGetColumns(Tensor<T> &Input, size_t StartIndex,
                                size_t EndIndex) {
  [[maybe_unused]] auto Difference =
      static_cast<long>(EndIndex) - static_cast<long>(StartIndex);
  assert(Difference >= 0);
  auto ColumnsFromStart = EndIndex - StartIndex;
  assert(Input.Type == TensorType::Matrix);
  assert(EndIndex < Input.Dimensions[1]);
  auto ColumnLength = Input.Strides[1];

  Tensor<T> NewTensor(
      ArenaArray<size_t>({Input.Dimensions[0], ColumnsFromStart + 1}),
      ArenaArray<T>(&Input[StartIndex * Input.Strides[1]],
                    ((ColumnsFromStart + 1) * ColumnLength)));
  assert(((ColumnsFromStart + 1) * ColumnLength) == NewTensor.Data.Size);

  return NewTensor;
}

#endif