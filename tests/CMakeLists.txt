cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("tests")

include_directories("../include")
#set(Caffe_DIR "/Users/worm004/caffe/build/install/share/Caffe")
set(Caffe_DIR "/Users/worm004/Projects/caffe-yolo/build/install/share/Caffe")
find_package(Caffe)
find_package(OpenCV)
add_definitions(-DCPU_ONLY)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(parser parser.cpp)
target_link_libraries(parser net004 ${OpenCV_LIBRARIES})

add_executable(net_test net_test.cpp)
target_link_libraries(net_test net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(detect_test detect_test.cpp)
target_link_libraries(detect_test net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
