set(EMULATOR_EXECUTABLE_NAME "emulator")

if(!UNIX)
    message(ERROR "Emulator runtime supports UNIX systems only")
    return()
endif()

add_executable(${EMULATOR_EXECUTABLE_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/emulator_main.cpp
)

target_link_libraries(${EMULATOR_EXECUTABLE_NAME} PUBLIC
    emulator_host
)

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
