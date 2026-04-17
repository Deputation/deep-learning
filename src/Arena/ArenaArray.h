#ifndef ARENA_ARRAY_H
#define ARENA_ARRAY_H

#include "Arena.h"
#include "IntegerGeneratorWrapper.h"

#include <cstddef>
#include <string.h>

template <typename T> class ArenaArray {
public:
  T *Data;
  size_t Size;
  size_t FreeIndex;

  ArenaArray() : Data(nullptr), Size(0) {}

  explicit ArenaArray(T *Data, size_t Size) : Data(Data), Size(Size) {}

  explicit ArenaArray(size_t Size, Arena *AllocationArena = &GeneralArena)
      : Size(Size) {
    Data = new (AllocationArena->Allocate(sizeof(T) * Size)) T[Size];
    FreeIndex = 0;
  }

  ArenaArray(std::initializer_list<T> Elements,
             Arena *AllocationArena = &GeneralArena)
      : Size(Elements.size()) {
    Data = new (AllocationArena->Allocate(sizeof(T) * Size)) T[Size];

    auto i = 0;
    for (auto &Element : Elements) {
      Data[i] = Element;
      i++;
      FreeIndex++;
    }
  }

  ArenaArray<T> AllocateCopy() const {
    ArenaArray<T> Copy(Size);
    for (size_t i = 0; i < Size; i++) {
      Copy[i] = Data[i];
      Copy.FreeIndex = FreeIndex;
    }

    return Copy;
  }

  auto Push(T Object) {
    assert(FreeIndex < Size);

    auto Index = FreeIndex;
    FreeIndex++;
    Data[Index] = Object;
    return Index;
  }

  auto Pop() {
    assert(FreeIndex > 0);

    memset(&Data[FreeIndex - 1], 0, sizeof(T));
    FreeIndex--;
  }

  inline auto &operator[](size_t Index) {
    assert(Index < Size);

    return Data[Index];
  }

  inline const auto &operator[](size_t Index) const {
    assert(Index < Size);

    return Data[Index];
  }
};

#endif