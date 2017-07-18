cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project("tests")
find_package(OpenCV)

include_directories("../include")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_connections test_connections.cpp)
add_executable(test_layers test_layers.cpp)
add_executable(test_net test_net.cpp)
add_executable(test_forward test_forward.cpp)

target_link_libraries(test_connections net004)
target_link_libraries(test_layers net004)
target_link_libraries(test_net net004)
target_link_libraries(test_forward net004 ${OpenCV_LIBRARIES})
