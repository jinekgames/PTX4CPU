set(EMULATOR_LIB_NAME "EmulatorCore")

set(EMULATOR_UTILS_DIR ${EMULATOR_DIR}/utils)


set(translator_sources
    ${TRANSLATOR_DIR}/translator.cpp
    ${TRANSLATOR_DIR}/translator_interface.cpp
    ${PARSER_DIR}/parser.cpp
    ${EMULATOR_UTILS_DIR}/version.cpp
)

set(translator_includes
    ${TRANSLATOR_DIR}
    ${PARSER_DIR}
)


set(emulator_api_sources
    ${EMULATOR_API_DIR}/emulator_api.cpp
)

set(emulator_api_includes
    ${EMULATOR_API_DIR}
)


set(sources
    ${translator_sources}
    ${emulator_api_sources}
)

set(includes
    ${translator_includes}
    ${emulator_api_includes}
    ${EMULATOR_DIR}
    ${EMULATOR_UTILS_DIR}
    ${UTILS_DIR}/logger
)


include_directories(${includes})

add_library(emulatorlib SHARED ${sources})

set_target_properties(emulatorlib PROPERTIES OUTPUT_NAME ${EMULATOR_LIB_NAME})

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

target_compile_definitions(emulatorlib PRIVATE -DPROJECT_GIT_COMMIT="${GIT_HASH}")
