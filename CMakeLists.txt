cmake_minimum_required(VERSION 4.0)
project(deep-learning)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)

add_compile_options(-Wall -Wextra -Wpedantic)

add_subdirectory(src)
