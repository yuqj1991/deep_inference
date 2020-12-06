#Author         :yuqj
#Email          :BrixLab
#Last_update    :2020.12.02

cmake_minimum_required(VERSION 3.5)
project(onednn_inference)

SET(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)

add_definitions(-g -Wall -std=c++11)
set(INC_DIR /usr/local/include include)
set(LINK_DIR /usr/local/lib)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

set(Lib_DIR lib)
link_directories(${Lib_DIR})
set(onednn_lib dnnl)
link_libraries(${onednn_lib})
#find_package(OpenCV 3 REQUIRED)

include_directories(${INC_DIR})

option(DNNL_VERBOSE "allows oneDNN be verbose whenever DNNL_VERBOSE environment variable set to 1" ON)

add_executable(demo_inference
        include/
        source/liteopConvert.cpp
        source/layer_builder.cpp
        source/utils.cpp
        source/unit_test.cpp
        source/main.cpp)
        
#target_link_libraries(demo_inference ${OpenCV_LIBS})
target_link_libraries(demo_inference ${onednn_lib})