add_library(ArenaLibrary STATIC Arena.h ArenaArray.h ArenaArrayOperations.h ArenaWatcher.h GeneratorWrapper.h IntegerGeneratorWrapper.h Arena.cpp ArenaWatcher.cpp)

target_include_directories(ArenaLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})