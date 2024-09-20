
set(sources
    ${DTABLE_RUNNERS_DIR}/base.cpp
    ${DTABLE_RUNNERS_DIR}/math.cpp
    ${DTABLE_RUNNERS_DIR}/memory.cpp
    ${DTABLE_RUNNERS_DIR}/runner.h
)

add_library(runnersDispatchTable OBJECT ${sources})
