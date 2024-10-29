
set(sources
    ${DTABLE_RUNNERS_DIR}/base.cpp
    ${DTABLE_RUNNERS_DIR}/math.cpp
    ${DTABLE_RUNNERS_DIR}/memory.cpp
    ${DTABLE_RUNNERS_DIR}/runner.h
)

if(UNIX)
    set_source_files_properties(
        ${DTABLE_RUNNERS_DIR}/math.cpp
        ${DTABLE_RUNNERS_DIR}/memory.cpp
        PROPERTIES COMPILE_FLAGS "-Wno-format-security"
    )
endif()

add_library(runnersDispatchTable OBJECT ${sources})
