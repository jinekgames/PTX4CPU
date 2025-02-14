set(EMULATOR_LIB_NAME "EmulatorCore")

set(JSON_PARSER_DIR    ${3RDPARTY_DIR}/json)
set(EMULATOR_API_DIR   ${EMULATOR_DIR}/api)
set(EMULATOR_UTILS_DIR ${EMULATOR_DIR}/utils)
set(EXECUTOR_DIR       ${PARSER_DIR}/executor)
set(DTABLE_RUNNERS_DIR ${EXECUTOR_DIR}/runners)


set(emulator_sources
    ${EMULATOR_DIR}/emulator/emulator_interface.cpp
    ${EMULATOR_DIR}/emulator/emulator.cpp
    ${EMULATOR_DIR}/emulator/emulator.h
    ${EXECUTOR_DIR}/dispatch_table.cpp
    ${EXECUTOR_DIR}/executor.cpp
    ${EXECUTOR_DIR}/executor.h
    ${EXECUTOR_DIR}/instruction_runner.cpp
    ${EXECUTOR_DIR}/instruction_runner.h
    ${PARSER_DIR}/arg_parsers/arg_parsers.cpp
    ${PARSER_DIR}/arg_parsers/arg_parsers.h
    ${PARSER_DIR}/arg_parsers/json_parser/parser.h
    ${PARSER_DIR}/parser.cpp
    ${PARSER_DIR}/parser.h
    ${PARSER_DIR}/parser_types.h
    ${PARSER_DIR}/types/parser_data.cpp
    ${PARSER_DIR}/types/parser_data.h
    ${PARSER_DIR}/types/ptx_function.cpp
    ${PARSER_DIR}/types/ptx_function.h
    ${PARSER_DIR}/types/ptx_types.cpp
    ${PARSER_DIR}/types/ptx_types.h
    ${PARSER_DIR}/types/virtual_var.cpp
    ${PARSER_DIR}/types/virtual_var.h
)

set(emulator_includes
    ${EXECUTOR_DIR}
    ${JSON_PARSER_DIR}/include
    ${PARSER_DIR}
)


set(emulator_api_sources
    ${EMULATOR_API_DIR}/emulator_api.cpp
    ${EMULATOR_API_DIR}/export/emulator_api.h
    ${EMULATOR_API_DIR}/export/emulator/emulator_interface.h
    ${EMULATOR_API_DIR}/export/utils/base_types.h
    ${EMULATOR_API_DIR}/export/utils/result.h
)

set(emulator_api_includes
    ${EMULATOR_API_DIR}/export
)


if(UNIX)
    set_source_files_properties(
        ${EMULATOR_API_DIR}/emulator_api.cpp
        ${EMULATOR_DIR}/emulator/emulator.cpp
        ${EXECUTOR_DIR}/executor.cpp
        ${PARSER_DIR}/parser.cpp
        ${PARSER_DIR}/types/ptx_function.cpp
        ${PARSER_DIR}/types/ptx_types.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-format-security"
    )
endif()


set(sources
    ${emulator_sources}
    ${emulator_api_sources}
)

set(includes
    ${emulator_includes}
    ${emulator_api_includes}
    ${EMULATOR_DIR}
    ${EMULATOR_UTILS_DIR}
    ${UTILS_DIR}
)


include_directories(${includes})

add_subdirectory(${DTABLE_RUNNERS_DIR})
add_subdirectory(${EMULATOR_UTILS_DIR})

set(objs
    $<TARGET_OBJECTS:runnersDispatchTable>
    $<TARGET_OBJECTS:emulatorUtils>
)

add_library(emulatorLib SHARED ${sources} ${objs})

set_target_properties(emulatorLib PROPERTIES OUTPUT_NAME ${EMULATOR_LIB_NAME})

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

# Install API headers to build directory
add_custom_command(TARGET emulatorLib PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${EMULATOR_API_DIR}/export ${CMAKE_BINARY_DIR}/api)
