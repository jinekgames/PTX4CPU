set(EMULATOR_LIB_NAME "EmulatorCore")

set(JSON_PARSER_DIR    ${3RDPARTY_DIR}/json)
set(EMULATOR_UTILS_DIR ${EMULATOR_DIR}/utils)
set(EXECUTOR_DIR       ${PARSER_DIR}/executor)
set(DTABLE_RUNNERS_DIR ${EXECUTOR_DIR}/runners)


set(emulator_sources
    ${EXECUTOR_DIR}/dispatch_table.cpp
    ${EXECUTOR_DIR}/dispatch_table.h
    ${EXECUTOR_DIR}/executor.cpp
    ${EXECUTOR_DIR}/executor.h
    ${EXECUTOR_DIR}/instruction.cpp
    ${EXECUTOR_DIR}/instruction.h
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
    ${TRANSLATOR_DIR}/translator.cpp
    ${TRANSLATOR_DIR}/translator.h
    ${TRANSLATOR_DIR}/translator_interface.cpp
    ${TRANSLATOR_DIR}/translator_interface.h
)

set(emulator_includes
    ${EXECUTOR_DIR}
    ${JSON_PARSER_DIR}/include
    ${TRANSLATOR_DIR}
    ${PARSER_DIR}
)


set(emulator_api_sources
    ${EMULATOR_API_DIR}/emulator_api.cpp
    ${EMULATOR_API_DIR}/emulator_api.h
)

set(emulator_api_includes
    ${EMULATOR_API_DIR}
)


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

target_compile_definitions(emulatorLib PRIVATE -DPROJECT_GIT_COMMIT="${GIT_HASH}")
