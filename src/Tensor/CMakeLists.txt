add_library(TensorLibrary STATIC Tensor.h Tensor.cpp)

target_include_directories(TensorLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TensorLibrary ArenaLibrary)

add_subdirectory(Tests)