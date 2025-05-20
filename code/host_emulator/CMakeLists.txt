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

endif()
