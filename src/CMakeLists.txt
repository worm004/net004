cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("net004")
find_package(OpenCV)
include_directories("../include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${vecLib_INCLUDE_DIR})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(net004 STATIC Parser.cpp Net004.cpp Layers.cpp Blob.cpp im2col.cpp NetGame.cpp JsonParser.cpp
			BaseRun.cpp ForwardBackwardRun.cpp SaveRun.cpp DisplayRun.cpp ForwardTestRun.cpp InitRun.cpp UpdateRun.cpp
			BaseLayer.cpp ConvLayer.cpp FCLayer.cpp ActivityLayer.cpp LossLayer.cpp DataLayer.cpp PoolLayer.cpp LRNLayer.cpp SplitLayer.cpp ConcatLayer.cpp BNLayer.cpp ScaleLayer.cpp EltwiseLayer.cpp SoftmaxLayer.cpp ReshapeLayer.cpp ProposalLayer.cpp RoipoolLayer.cpp CropLayer.cpp DConvLayer.cpp)

target_link_libraries(net004 ${vecLib_LINKER_LIBS} ${OpenCV_LIBRARIES} )
