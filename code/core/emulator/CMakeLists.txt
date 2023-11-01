set(EMULATOR_LIB_NAME "EmulatorCore")


set(translator_sources
    ${TRANSLATOR_DIR}/translator.cpp
)

set(translator_includes
    ${TRANSLATOR_DIR}
)


set(emulator_api_sources
    ${EMULATOR_API_DIR}/emulator_api.cpp
    ${EMULATOR_API_DIR}/emulator_api.h
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
)


include_directories(${includes})

add_library(emulatorlib SHARED ${sources})

set_target_properties(emulatorlib PROPERTIES OUTPUT_NAME ${EMULATOR_LIB_NAME})
