#include "ArenaWatcher.h"

ArenaWatcher::ArenaWatcher(Arena &Arena) : WatchedArena(Arena) {
  State = WatchedArena.SaveState();
}

ArenaWatcher::~ArenaWatcher() { WatchedArena.RestoreState(State); }