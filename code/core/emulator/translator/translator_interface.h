#pragma once

#include <string>

#include <parser/parser_types.h>
#include <utils/result.h>
#include <utils/base_types.h>


namespace PTX4CPU {

struct ITranslator {

    ITranslator();

    /**
     * Execute named function from the loaded PTX.
     *
     * The kernel, specified by the `funcName` and `args` count and types
     * will be executed in a grid the size `gridSize` with `args` input
     * arguments.
     * (gridSize.x * gridSize.y * gridSize.z) execution threads will be created.
     *
     * @param funcName  compiled name of function
     * @param args      list of execution args
     * @param gridSize  size of grid
     *
     * @return Result
    */
    virtual Result ExecuteFunc(const std::string& funcName, Types::PTXVarList& args,
                               const uint3_32& gridSize) = 0;

};

};  // namespace PTX4CPU
