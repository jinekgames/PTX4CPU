set(3RDPARTY_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(CORE_DIR         ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(RUNTIME_LIB_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/host_emulator)
set(TESTS_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(UTILS_DIR        ${CORE_DIR}/../utils)
set(EMULATOR_DIR     ${CORE_DIR}/emulator)
set(INTERFACES_DIR   ${CORE_DIR}/interfaces)
set(EMULATOR_API_DIR ${EMULATOR_DIR}/api)
set(PARSER_DIR       ${EMULATOR_DIR}/parser)
set(CMD_PARSER_DIR   ${UTILS_DIR}/cmd_parser)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(${EMULATOR_DIR})

set(EMULATOR_CMD_DIR ${INTERFACES_DIR}/emulator_cmd)

if(BUILD_EMULATOR_CMD)
    add_subdirectory(${CMD_PARSER_DIR})
    add_subdirectory(${EMULATOR_CMD_DIR})
    add_subdirectory(${RUNTIME_LIB_DIR})
endif()

if(BUILD_TESTS)
    add_subdirectory(${TESTS_DIR})
endif()

# Getting a commit hash

if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
   execute_process(
       COMMAND git rev-parse HEAD
       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
       OUTPUT_VARIABLE GIT_HASH
       OUTPUT_STRIP_TRAILING_WHITESPACE
   )
endif()
if(NOT GIT_HASH)
   set(GIT_HASH "NO_COMMIT_INFORMATION")
endif()
string(STRIP ${GIT_HASH} GIT_HASH)
message(STATUS "Building from commit: ${GIT_HASH}")

add_compile_definitions(PROJECT_GIT_COMMIT="${GIT_HASH}")
