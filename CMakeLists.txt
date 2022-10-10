cmake_minimum_required(VERSION 3.23)
project(AI_class)

set(CMAKE_CXX_STANDARD 14)

add_executable(AI_class
        main.cpp
        tool.cpp )

add_executable(test
        test_main.cpp
)

add_executable(class_work
        class_work/1006/1006_back_propagation.cpp class_work/1006/1006_mmt.cpp)