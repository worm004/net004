cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
include(cmake/FindvecLib.cmake)
project(net004)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

#add_subdirectory(src)
#add_subdirectory(example)


add_subdirectory(src2)
add_subdirectory(example2)
