set(TRANSLATOR_CMD_EXENAME "TranslatorCmd")


set(sources
    ./translator_cmd.cpp
)

set(includes
    ${EMULATOR_API_DIR}
    ${EMULATOR_DIR}
    ${TRANSLATOR_DIR}
)


list(APPEND sources
    ${UTILS_DIR}/cmd_parser/cmd_parser.cpp
)

list(APPEND includes
    ${UTILS_DIR}
)


include_directories(${includes})

add_executable(translator_cmd_exe ${sources})

set_target_properties(translator_cmd_exe PROPERTIES OUTPUT_NAME ${TRANSLATOR_CMD_EXENAME})

add_dependencies(translator_cmd_exe emulatorlib)
target_link_libraries(translator_cmd_exe PUBLIC emulatorlib)
