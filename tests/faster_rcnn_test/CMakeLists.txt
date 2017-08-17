cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("tests")

include_directories("../../include")
set(Caffe_DIR "/Users/worm004/Projects/py-faster-rcnn/caffe-fast-rcnn/build/install/share/Caffe")
find_package(Caffe)
find_package(OpenCV)

set(PYTHON_LIBRARY "/usr/local/Cellar/python/2.7.9/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
find_package(PythonInterp 2.7)
find_package(PythonLibs 2.7)

add_definitions(-DCPU_ONLY)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(faster_rcnn_test faster_rcnn_test.cpp faster_rcnn_process.cpp)
target_link_libraries(faster_rcnn_test net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
