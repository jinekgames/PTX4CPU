set(RUNTIME_LIB_NAME "emulator_host")

if(UNIX)
    add_library(${RUNTIME_LIB_NAME} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime_def.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/event_handler.cpp
    )

    target_include_directories(${RUNTIME_LIB_NAME} PUBLIC
        ${UTILS_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EMULATOR_API_DIR}
        ${EMULATOR_API_DIR}/export
    )

    target_link_libraries(${RUNTIME_LIB_NAME} PUBLIC
        emulatorLib
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
endif()
