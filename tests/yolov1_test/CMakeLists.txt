cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("yolov1_test")

include_directories("../../include")
set(Caffe_DIR "/Users/worm004/Projects/caffe-yolo/build/install/share/Caffe")
find_package(Caffe)
find_package(OpenCV)
add_definitions(-DCPU_ONLY)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov1_test yolov1_test.cpp)
target_link_libraries(yolov1_test net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
