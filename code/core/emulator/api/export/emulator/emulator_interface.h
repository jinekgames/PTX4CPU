#pragma once

#include <string>

#include "../utils/api_types.h"
#include "../utils/base_types.h"
#include "../utils/result.h"


namespace PTX4CPU {

struct IEmulator {

    IEmulator();
    virtual ~IEmulator() = default;

    /**
     * Execute named function from the loaded PTX.
     *
     * The kernel, specified by the `funcName` and `args` count and types
     * will be executed in a grid the size `gridSize` with `args` input
     * arguments.
     * (gridSize.x * gridSize.y * gridSize.z) execution threads will be
     * created.
     *
     * @param funcName compiled name of function
     * @param pArgs    list of execution args
     * @param gridSize size of grid
     *
     * @return Result
    */
    virtual Result ExecuteFunc(
        const std::string& funcName,
        PtxExecArgs pArgs,
        const BaseTypes::uint3_32& gridSize) = 0;

    /**
     * Retrieves the descriptor of a kernel with the given name.
     * Retrived descriptor is used by other APIs.
     * PTX should be successfully loaded into the Emulator.
     * Name should be unique across all kernels in the loaded PTX.
     *
     * @param name        Name of the kernel to get the descriptor of.
     * @param pDescriptor Pointer to the descriptor where the result will be
     * put.
     *
     * @return Result
     */
    virtual Result GetKernelDescriptor(
        const std::string& name,
        PtxFuncDescriptor* pDescriptor) const = 0;

};

};  // namespace PTX4CPU
