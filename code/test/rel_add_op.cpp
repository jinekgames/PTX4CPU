#include "rel_add_op.h"

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <vector>


#ifndef TESTS_EXT_DIR
#error "Empty tests directory compile-time constant"
#endif

constexpr auto PTX_FILE_PATH = TESTS_EXT_DIR "/cuda_ptx_samples/rel_add_op.ptx";
constexpr auto KERNEL_NAME   = "_Z9addKernelPiPKiS1_";


namespace TestCase {

namespace Runtime {

std::string RelAddOp::Name() {
    return "[Runtime] rel_add_op (vadd)";
}

std::string RelAddOp::Description() {
    return "Testing rel_add_op (vadd) PTX with emulated CUDA Runtime";
}

PTX4CPU::Result RelAddOp::Run() {

    // Read PTX file

    auto ptxSource = ReadFile(PTX_FILE_PATH);

    if (ptxSource.empty()) {
        return { "Failed to read source PTX file " +
                 std::string(PTX_FILE_PATH) };
    }


    // Create Emulator object

    PTX4CPU::IEmulator* pEmulator = nullptr;
    EMULATOR_CreateEmulator(&pEmulator, ptxSource);
    ptxSource.clear();

    if (pEmulator == nullptr) {
        return { "Failed to create Emulator object" };
    }


    // Setup Runtime arguments (this emulates arguments passed by the CUDA Runtime)

    constexpr size_t arraySize = 5;

    using TestType = uint32_t;

    std::vector<TestType> arr[3];
    arr[0].resize(arraySize); // the output array. keep trash data
    arr[1] = { 1, 2, 3, 4, 5, };
    arr[2] = { 1, 1, 1, 1, 1, };

    const std::vector<TestType*> argsList = {
        arr[0].data(),
        arr[1].data(),
        arr[2].data(),
    };

    const std::vector<TestType* const*> modArgsList = {
        &argsList[0],
        &argsList[1],
        &argsList[2],
    };

    void** ppArgs = (void**)modArgsList.data();


    // Retrieve kernel descriptor

    PTX4CPU::PtxFuncDescriptor kernelDescriptor = PTX4CPU_NULL_HANDLE;
    auto result =
        pEmulator->GetKernelDescriptor(KERNEL_NAME, &kernelDescriptor);

    if (!result) {
        return result;
    }


    // Process PTX arguments

    PTX4CPU::PtxExecArgs ptxArgs = PTX4CPU_NULL_HANDLE;
    EMULATOR_CreateArgs(&ptxArgs, kernelDescriptor, ppArgs);

    if (!ptxArgs) {
        return { "Failed to process kernel execution arguments" };
    }


    // Run kernel

    constexpr BaseTypes::uint3_32 gridSize = { arraySize, 1, 1 };

    result = pEmulator->ExecuteFunc(KERNEL_NAME, ptxArgs, gridSize);

    if (!result) {
        return result;
    }


    // Clean up

    EMULATOR_DestroyArgs(ptxArgs);
    ptxArgs = PTX4CPU_NULL_HANDLE;

    EMULATOR_DestroyEmulator(pEmulator);
    pEmulator = nullptr;


    // Check values

    std::vector<TestType> expectedArr;
    for (size_t i = 0; i < arraySize; ++i) {
        const auto value = arr[1][i] + arr[2][i];
        expectedArr.push_back(value);
    }

    bool success = true;
    std::cout << "Check:" << std::endl;
    for (size_t i = 0; i < gridSize.x; ++i) {
        std::cout << "[" << i << "]" << " expected: " << expectedArr[i]
                                     << " got: "      << arr[0][i]
                                     << std::endl;
        success &= (expectedArr[i] == arr[0][i]);
    }

    if (!success) {
        return { "Invalid output values" };
    }

    return {};
}

}  // namespace Runtime

}  // namespace TestCase
