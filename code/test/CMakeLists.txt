set(tests
    rel_add_op.cpp
    rel_add_op.h)


set(sources
    tests.cpp
    tests.h
    utils.cpp
    utils.h)


set(includes
    ${EMULATOR_API_DIR}/export)


add_executable(integration_tests ${sources} ${tests})

target_include_directories(integration_tests PUBLIC ${includes})

add_dependencies(integration_tests emulatorLib)
target_link_libraries(integration_tests PUBLIC emulatorLib)


set(TESTS_EXT_DIR   ${CMAKE_BINARY_DIR}/tests)
set(PTX_SAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ext)


# Copy test files to build directory
add_custom_command(TARGET integration_tests PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${PTX_SAMPLES_DIR}/cuda_ptx_samples
                   ${TESTS_EXT_DIR}/cuda_ptx_samples)

target_compile_definitions(integration_tests PRIVATE
                           -DTESTS_EXT_DIR="${TESTS_EXT_DIR}")
