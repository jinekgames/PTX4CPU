cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CUDA_X86_EMULATOR)


if (!WIN32)
    message(ERROR "Emulator supports WIN32 systems only")
endif()


set(CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/code)

add_subdirectory(${CODE_DIR})