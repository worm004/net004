cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("net004")

include_directories("../include")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(net004 STATIC Connections.cpp Layers.cpp Net004.cpp Blob.cpp
		BaseLayer.cpp ConvLayer.cpp ConcatLayer.cpp PoolLayer.cpp LossLayer.cpp LRNLayer.cpp FCLayer.cpp DataLayer.cpp ActivityLayer.cpp
		im2col.cpp)
