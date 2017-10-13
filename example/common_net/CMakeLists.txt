cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("parser")

include_directories("../../include")
set(Caffe_DIR "/Users/worm004/caffe/build/install/share/Caffe")
find_package(Caffe)
find_package(OpenCV)
add_definitions(-DCPU_ONLY)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(model_cvt model_cvt.cpp caffe_parser.cpp)
target_link_libraries(model_cvt net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(net_test net_test.cpp)
target_link_libraries(net_test net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(caffe_show caffe_show.cpp)
target_link_libraries(caffe_show net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(net_train net_train.cpp)
target_link_libraries(net_train net004 ${OpenCV_LIBRARIES})

add_executable(caffe_train caffe_train.cpp)
target_link_libraries(caffe_train net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
