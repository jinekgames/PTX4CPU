cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 23)

project(CUDA_CPU_EMULATOR
        VERSION 0.1
        DESCRIPTION "Emulator for running CUDA code on CPU"
        LANGUAGES CXX)


if (!WIN32)
    message(ERROR "Emulator supports WIN32 systems only")
endif()


message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_definitions(DEBUG_BUILD)
endif()


set(CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/code)

add_subdirectory(${CODE_DIR})