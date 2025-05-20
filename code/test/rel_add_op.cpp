#include "rel_add_op.h"

#include "utils.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>


constexpr auto PTX_FILE_PATH = "/cuda_ptx_samples/rel_add_op.ptx";
constexpr auto KERNEL_NAME   = "_Z9addKernelPiPKiS1_";


namespace TestCase {

namespace Runtime {

std::string RelAddOp::Name() const {
    return kNamePrefix + std::string{kName};
}

std::string RelAddOp::Description() const {
    return std::string{"Testing "} + kName + " PTX with emulated CUDA Runtime";
}

PTX4CPU::Result RelAddOp::Run(const std::string& testAssetPath) const {

    // Read PTX file

    auto ptxSource = ReadFile(testAssetPath + PTX_FILE_PATH);

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

    // arg arrays
    std::array<TestType, arraySize> out, left, right;
    for (TestType i = 0; i < arraySize; ++i)  left[i]  = i + 1;
    for (TestType i = 0; i < arraySize; ++i)  right[i] = 10;

    std::array<TestType, arraySize> expectedOut, expectedLeft, expectedRight;
    expectedLeft  = left;
    expectedRight = right;
    for (size_t i = 0; i < arraySize; ++i)  expectedOut[i] = left[i] + right[i];

    // real args - pointers to arrays
    const std::vector<TestType*> argsList = {
        out.data(),
        left.data(),
        right.data(),
    };

    // runtime args - pointers to each real arg
    const std::vector<TestType* const*> modArgsList = {
        &argsList[0],
        &argsList[1],
        &argsList[2],
    };

    // real runtime arg - pointer to array of of runtime args
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

    CudaTypes::uint3 gridSize = { arraySize, 1, 1 };

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

    bool success = true;

    std::cout << std::endl << "Check:" << std::endl;

    const auto printCompare = [](
        std::string first, std::string second, std::string third, int i = -1) {

        std::cout << std::setw(0);
        constexpr size_t shift = 25;
        if (i >= 0)  first = "[" + std::to_string(i) + "] " + first;

        std::cout << std::setw(shift) << first;
        std::cout << std::setw(shift) << second;
        std::cout << std::setw(shift) << third;

        std::cout << std::setw(0) << std::endl;
    };

    printCompare("out", "left", "right");
    for (size_t i = 0; i < gridSize.x; ++i) {

        std::stringstream outStr;
        outStr << "expected: " << expectedOut[i]
               << " got: "     << out[i];
        std::stringstream leftStr;
        leftStr << "expected: " << expectedLeft[i]
                << " got: "     << left[i];
        std::stringstream rightStr;
        rightStr << "expected: " << expectedRight[i]
                 << " got: "     << right[i];

        printCompare(outStr.str(), leftStr.str(), rightStr.str(), i);

        success &= (expectedOut[i]   == out[i]);
        success &= (expectedLeft[i]  == left[i]);
        success &= (expectedRight[i] == right[i]);
    }

    if (!success) {
        return { "Invalid output values" };
    }

    return {};
}

}  // namespace Runtime

}  // namespace TestCase
