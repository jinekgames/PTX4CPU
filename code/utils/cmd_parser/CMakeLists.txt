set(sources
    ${CMD_PARSER_DIR}/cmd_parser.h
    ${CMD_PARSER_DIR}/cmd_parser.cpp
)

add_library(cmdLineParser OBJECT ${sources})
