cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("faster_rcnn")

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

#add_executable(caffe_show_yolo caffe_show.cpp)
#target_link_libraries(caffe_show_yolo ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(model_cvt_faster_rcnn model_cvt.cpp caffe_parser.cpp faster_rcnn_tool.cpp)
target_link_libraries(model_cvt_faster_rcnn ${Caffe_LIBRARIES} net004 ${OpenCV_LIBRARIES})

#add_executable(net_test_yolo net_test.cpp)
#target_link_libraries(net_test_yolo net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(net_test_faster_rcnn net_test.cpp faster_rcnn_tool.cpp)
target_link_libraries(net_test_faster_rcnn net004 ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
