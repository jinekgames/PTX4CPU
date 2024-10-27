cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 23)
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++23 -fPIC")
endif()

project(PTX4CPU
        VERSION 0.2.0
        DESCRIPTION "Emulator for running CUDA code on CPU"
        LANGUAGES CXX)


# @todo implementation: maybe move some of these into a runtime config
option(COMPILE_SAFE_CHECKS        "If set, the additional data checking will be compiled" ON)
option(USE_DEBUG_FEATURES         "Compile debug features even in non-debug build"        ON)
option(EXTENDED_VARIABLES_LOGGING "Build extended debug logging of variables content"     OFF)

option(BUILD_EMULATOR_CMD         "Build ptx emulator comand-line interface"              ON)
option(BUILD_TESTS                "Build autotests executable"                            OFF)


if(BUILD_TESTS AND NOT BUILD_EMULATOR_CMD)
    message(ERROR "Command-line tool must be built for tests")
endif()

if(!WIN32 OR !UNIX)
    message(ERROR "Emulator supports WIN32 and UNIX systems only")
endif()


message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR
   CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR
   USE_DEBUG_FEATURES)
    add_compile_definitions(DEBUG_BUILD)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    add_compile_definitions(SYSTEM_ARCH_64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    add_compile_definitions(SYSTEM_ARCH_32)
endif()

add_compile_definitions(PTX4CPU_VERSION="${CMAKE_PROJECT_VERSION}")

if(COMPILE_SAFE_CHECKS)
    add_compile_definitions(COMPILE_SAFE_CHECKS)
endif()

if(EXTENDED_VARIABLES_LOGGING)
    add_compile_definitions(EXTENDED_VARIABLES_LOGGING)
endif()


set(CODE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/code)

add_subdirectory(${CODE_DIR})