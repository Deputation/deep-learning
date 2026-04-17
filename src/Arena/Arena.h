#ifndef ARENA_H
#define ARENA_H

#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <mutex>

#define THREAD_SAFE_ARENA 0

struct ArenaState {
  size_t AllocatedBytes;
  size_t AllocationsCounter;
};

class Arena {
public:
#if THREAD_SAFE_ARENA
  std::mutex AllocationMutex;
#endif

  void *MemoryPool;
  size_t MemoryPoolSize;
  size_t AllocatedBytes;
  size_t AllocationsCounter;

  Arena(size_t StartingSize = 4'294'967'296);

  ~Arena();

  void *Allocate(size_t AllocationSize);
  ArenaState SaveState();
  void RestoreState(ArenaState &State);
};

static Arena GeneralArena;

#endif