#ifndef ARENA_WATCHER_H
#define ARENA_WATCHER_H

#include "Arena.h"

class ArenaWatcher {
public:
  Arena &WatchedArena;
  ArenaState State;
  ArenaWatcher(Arena &Arena);
  ~ArenaWatcher();
};

#endif