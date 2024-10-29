set(sources
    ${CODE_DIR}/utils/logger/logger.h
    ${EMULATOR_UTILS_DIR}/base_types.h
    ${EMULATOR_UTILS_DIR}/iterator.h
    ${EMULATOR_UTILS_DIR}/iterator.inl
    ${EMULATOR_UTILS_DIR}/result.h
    ${EMULATOR_UTILS_DIR}/string_ext/string_iterator.cpp
    ${EMULATOR_UTILS_DIR}/string_ext/string_iterator.h
    ${EMULATOR_UTILS_DIR}/string_ext/string_iterator.inl
    ${EMULATOR_UTILS_DIR}/string_ext/string_types.h
    ${EMULATOR_UTILS_DIR}/string_utils.h
    ${EMULATOR_UTILS_DIR}/version.cpp
    ${EMULATOR_UTILS_DIR}/version.h
)

add_library(emulatorUtils OBJECT ${sources})
