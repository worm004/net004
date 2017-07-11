cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("net004")

include_directories("../include")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(net004 STATIC Connections.cpp)
