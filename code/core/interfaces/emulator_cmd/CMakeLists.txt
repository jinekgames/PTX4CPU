set(TRANSLATOR_CMD_EXENAME "EmulatorCmd")


set(sources
    ${EMULATOR_CMD_DIR}/logo_ascii.h
    ${EMULATOR_CMD_DIR}/translator_cmd.cpp
)

set(includes
    ${EMULATOR_API_DIR}
    ${EMULATOR_DIR}
    ${TRANSLATOR_DIR}
)


list(APPEND includes
    ${CMD_PARSER_DIR}
    ${UTILS_DIR}
)


include_directories(${includes})

set(objs
    $<TARGET_OBJECTS:cmdLineParser>
)

add_executable(translator_cmd_exe ${sources} ${objs})

set_target_properties(translator_cmd_exe PROPERTIES OUTPUT_NAME ${TRANSLATOR_CMD_EXENAME})

add_dependencies(translator_cmd_exe emulatorLib)
target_link_libraries(translator_cmd_exe PUBLIC emulatorLib)
