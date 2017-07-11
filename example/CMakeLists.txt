cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("tests")

include_directories("../include")
add_executable(test_connections test_connections.cpp)
target_link_libraries(test_connections net004)
