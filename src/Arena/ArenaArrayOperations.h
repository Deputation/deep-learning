#ifndef ARENA_ARRAY_OPERATIONS_H
#define ARENA_ARRAY_OPERATIONS_H

#include "ArenaArray.h"

template <typename T> static void Shuffle(ArenaArray<T> &Array) {
  auto &IndexGenerator = IntegerGeneratorWrapper<size_t>::GetSingleton();

  for (size_t i = 0; i < Array.Size; i++) {
    auto RandomIndex = IndexGenerator.Sample() % (Array.Size - 1);
    std::swap(Array.Data[i], Array.Data[RandomIndex]);
  }
}

#endif