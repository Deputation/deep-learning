#include "Arena.h"

Arena::Arena(size_t StartingSize)
    : MemoryPool(malloc(StartingSize)), MemoryPoolSize(StartingSize),
      AllocatedBytes(0), AllocationsCounter(0) {}

Arena::~Arena() { free(MemoryPool); }

void *Arena::Allocate(size_t AllocationSize) {
#if THREAD_SAFE_ARENA
  std::lock_guard ArenaGuard(AllocationMutex);
#endif

  if ((AllocatedBytes + AllocationSize) >= MemoryPoolSize) {
    assert(false && "Memory pool limit reached.");
    return nullptr;
  }

  size_t AllocationOffset = AllocatedBytes;
  void *MemoryAddress =
      reinterpret_cast<uint8_t *>(MemoryPool) + AllocationOffset;
  AllocatedBytes += AllocationSize;
  AllocationsCounter++;

  return MemoryAddress;
}

ArenaState Arena::SaveState() { return {AllocatedBytes, AllocationsCounter}; }

void Arena::RestoreState(ArenaState &State) {
#if THREAD_SAFE_ARENA
  std::lock_guard ArenaGuard(AllocationMutex);
#endif

  AllocatedBytes = State.AllocatedBytes;
  AllocationsCounter = State.AllocationsCounter;
}