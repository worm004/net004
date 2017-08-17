cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
include(cmake/FindvecLib.cmake)
project(net004)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(example)
#add_subdirectory(tools)
#add_subdirectory(tests)
