cmake_minimum_required(VERSION 3.25)
project(yolo_and_detection_together)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 23)

add_executable(yolo_and_detection_together main.cpp)
target_link_libraries(yolo_and_detection_together ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

