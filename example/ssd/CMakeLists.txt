cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("ssd")

include_directories("../../include")
set(Caffe_DIR "/Users/worm004/Projects/caffe-ssd/caffe/build/install/share/Caffe")
find_package(Caffe)
find_package(OpenCV)
add_definitions(-DCPU_ONLY)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(caffe_show_ssd caffe_show.cpp)
target_link_libraries(caffe_show_ssd ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(model_cvt_ssd model_cvt.cpp caffe_parser.cpp)
target_link_libraries(model_cvt_ssd ${Caffe_LIBRARIES} net004 ${OpenCV_LIBRARIES})

add_executable(net_test_ssd net_test.cpp)
target_link_libraries(net_test_ssd net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
