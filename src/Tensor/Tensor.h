#ifndef TENSOR_H
#define TENSOR_H

#include <ArenaArray.h>
#include <assert.h>
#include <cstdlib>
#include <functional>

enum class TensorType : unsigned char { Scalar, Vector, Matrix, Tensor };

template <typename T> class Tensor {
private:
  size_t ComputeDataSize() {
    size_t Result = 1;
    for (size_t i = 0; i < Dimensions.Size; i++) {
      Result *= Dimensions[i];
    }
    return Result;
  }

public:
  using TensorElementType = T;

  ArenaArray<size_t> Dimensions;
  ArenaArray<size_t> Strides;
  ArenaArray<T> Data;
  TensorType Type;

  Tensor() {}

  Tensor(T Initializer) : Dimensions({1, 1}), Data(ArenaArray<T>(1)) {
    assert(this->Dimensions.Size >= 2);
    Data[0] = Initializer;
    ComputeStrides();
  }

  Tensor(ArenaArray<size_t> Dimensions) : Dimensions(Dimensions) {
    assert(this->Dimensions.Size >= 2);

    Data = ArenaArray<T>(ComputeDataSize());
    for (size_t i = 0; i < Data.Size; i++) {
      Data[i] = static_cast<T>(0);
    }

    ComputeStrides();
  }

  Tensor(ArenaArray<size_t> Dimensions, T Initializer)
      : Dimensions(Dimensions) {
    assert(this->Dimensions.Size >= 2);

    Data = ArenaArray<T>(ComputeDataSize());
    for (size_t i = 0; i < Data.Size; i++) {
      Data[i] = Initializer;
    }

    ComputeStrides();
  }

  Tensor(ArenaArray<size_t> Dimensions, std::function<T()> Initializer)
      : Dimensions(Dimensions) {
    assert(this->Dimensions.Size >= 2);

    Data = ArenaArray<T>(ComputeDataSize());
    for (size_t i = 0; i < Data.Size; i++) {
      Data[i] = Initializer();
    }

    ComputeStrides();
  }

  Tensor(ArenaArray<size_t> Dimensions, ArenaArray<T> Initializer)
      : Dimensions(Dimensions), Data(Initializer) {
    assert(this->Dimensions.Size >= 2);

    [[maybe_unused]] auto DataSize = ComputeDataSize();
    assert(Initializer.Size == DataSize);

    ComputeStrides();
  }

  void ComputeStrides() {
    Strides = ArenaArray<size_t>(Dimensions.Size);

    if (Dimensions.Size == 2 && (Dimensions[0] == 1 || Dimensions[1] == 1)) {
      for (size_t i = 0; i < Dimensions.Size; i++) {
        Strides[i] = 1;
      }

      if (Dimensions[0] == 1 && Dimensions[1] == 1) {
        Type = TensorType::Scalar;

        return;
      }

      Type = TensorType::Vector;

      return;
    }

    // Incrementing the row by 1 lets you access the next column's element
    Strides[0] = 1;

    // Compute the next strides
    for (size_t i = 1; i < Strides.Size; i++) {
      // In a matrix, Strides[1] will then indicate how many elements ahead the
      // next element in a row is
      auto Result = 1;
      for (size_t j = 0; j < i; j++) {
        Result *= Dimensions[j];
      }
      Strides[i] = Result;
    }

    if (Dimensions.Size == 2) {
      Type = TensorType::Matrix;

      return;
    }

    Type = TensorType::Tensor;
  }

  bool StridesMatch(const Tensor<T> &Right) const {
    if (Strides.Size != Right.Strides.Size) {
      assert(false);
      return false;
    }

    for (size_t i = 0; i < Strides.Size; i++) {
      if (Right.Strides[i] != Strides[i]) {
        assert(false);
        return false;
      }
    }

    return true;
  }

  bool DimensionsMatch(const Tensor<T> &Right) const {
    if (Dimensions.Size != Right.Dimensions.Size) {
      assert(false);
      return false;
    }

    for (size_t i = 0; i < Dimensions.Size; i++) {
      if (Right.Dimensions[i] != Dimensions[i]) {
        assert(false);
        return false;
      }
    }

    return StridesMatch(Right);
  }

  inline size_t ComputeIndex(size_t Row, size_t Column) const {
    return (Row * Strides[0]) + (Column * Strides[1]);
  }

  inline size_t
  ComputeIndexNDimensional(std::initializer_list<size_t> Indices) const {
    assert(Indices.size() == Dimensions.Size);

    size_t ComputedIndex = 0;
    size_t StrideIndex = 0;
    for (const size_t &Index : Indices) {
      assert(Index < Dimensions[StrideIndex]);

      ComputedIndex += Index * Strides[StrideIndex];
      StrideIndex++;
    }

    return ComputedIndex;
  }

  inline T &operator[](std::initializer_list<size_t> Indices) {
    return Data[ComputeIndexNDimensional(Indices)];
  }

  inline const T &operator[](std::initializer_list<size_t> Indices) const {
    return Data[ComputeIndexNDimensional(Indices)];
  }

  inline T &operator[](size_t Row, size_t Column) {
    return Data[ComputeIndex(Row, Column)];
  }

  inline const T &operator[](size_t Row, size_t Column) const {
    return Data[ComputeIndex(Row, Column)];
  }

  inline T &operator[](size_t Index) { return Data[Index]; }

  inline const T &operator[](size_t Index) const { return Data[Index]; }
};

template <typename T> struct IsTensor : std::false_type {};

template <typename T> struct IsTensor<Tensor<T>> : std::true_type {};

#endif