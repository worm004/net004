cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("tests")

include_directories("../include")
set(Caffe_DIR "/Users/worm004/caffe/build/install/share/Caffe")
find_package(Caffe)
find_package(OpenCV)
add_definitions(-DCPU_ONLY)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cifar10_test cifar10_test.cpp)
target_link_libraries(cifar10_test ${Caffe_LIBRARIES} net004 ${OpenCV_LIBRARIES})

add_executable(vgg16_test vgg16_test.cpp)
target_link_libraries(vgg16_test ${Caffe_LIBRARIES} net004 ${OpenCV_LIBRARIES})

add_executable(parser parser.cpp)
target_link_libraries(parser net004 ${OpenCV_LIBRARIES})

add_executable(alexnet_test alexnet_test.cpp)
target_link_libraries(alexnet_test net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
