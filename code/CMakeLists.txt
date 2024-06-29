option(BUILD_EMULATOR_INTERFACE "Build main user intarface of program"       OFF)
option(BUILD_EMULATOR_CMD       "Build ptx translator comand-line interface" ON)


set(CORE_DIR         ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(EMULATOR_DIR     ${CORE_DIR}/emulator)
set(EMULATOR_API_DIR ${EMULATOR_DIR}/api)
set(PARSER_DIR       ${EMULATOR_DIR}/parser)
set(TRANSLATOR_DIR   ${EMULATOR_DIR}/translator)
set(INTERFACES_DIR   ${CORE_DIR}/interfaces)
set(UTILS_DIR        ${CORE_DIR}/../utils)
set(3RDPARTY_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(CMD_PARSER_DIR   ${UTILS_DIR}/cmd_parser)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# build emulator library
add_subdirectory(${EMULATOR_DIR})


if (BUILD_EMULATOR_INTERFACE)
    message(ERROR "Main user interface not implemented")
endif()


set(EMULATOR_CMD_DIR ${INTERFACES_DIR}/emulator_cmd)

if (BUILD_EMULATOR_CMD)
    add_subdirectory(${CMD_PARSER_DIR})
    add_subdirectory(${EMULATOR_CMD_DIR})
endif()