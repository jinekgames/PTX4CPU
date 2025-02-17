#include "rel_add_op_const.h"

#include "utils.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>


constexpr auto PTX_FILE_PATH = "/cuda_ptx_samples/rel_add_op_const.ptx";
constexpr auto KERNEL_NAME   = "_Z12taxpy_kerneliPi";


namespace TestCase {

namespace Runtime {

std::string RelAddOpConst::Name() const {
    return kNamePrefix + std::string{kName};
}

std::string RelAddOpConst::Description() const {
    return std::string{"Testing "} + kName + " PTX with emulated CUDA Runtime";
}

PTX4CPU::Result RelAddOpConst::Run(const std::string& testAssetPath) const {

    // Read PTX file

    std::cout << "1\n";

    auto ptxSource = ReadFile(testAssetPath + PTX_FILE_PATH);

    std::cout << "2\n";

    if (ptxSource.empty()) {
        return { "Failed to read source PTX file " +
                 std::string(PTX_FILE_PATH) };
    }

    std::cout << "3\n";


    // Create Emulator object

    PTX4CPU::IEmulator* pEmulator = nullptr;
    EMULATOR_CreateEmulator(&pEmulator, ptxSource);
    ptxSource.clear();

    std::cout << "4\n";

    if (pEmulator == nullptr) {
        return { "Failed to create Emulator object" };
    }


    // Setup Runtime arguments (this emulates arguments passed by the CUDA Runtime)

    constexpr size_t arraySize = 2;

    using TestType = uint32_t;

    // arg arrays
    std::array<TestType, arraySize> out, left;
    TestType right = 10;
    for (TestType i = 0; i < arraySize; ++i)  left[i] = i + 1;

    std::array<TestType, arraySize> expectedOut, expectedLeft;
    TestType expectedRight = right;
    expectedLeft = left;
    for (size_t i = 0; i < arraySize; ++i)  expectedOut[i] = left[i] + right;

    // real args - pointers to arrays
    const std::vector<TestType*> argsList = {
        out.data(),
        left.data(),
        &right,
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

    bool success = true;

    std::cout << std::endl << "Check:" << std::endl;

    const auto printCompare = [](
        std::string first, std::string second, int i = -1) {

        std::cout << std::setw(0);
        constexpr size_t shift = 25;
        if (i >= 0)  first = "[" + std::to_string(i) + "] " + first;

        std::cout << std::setw(shift) << first;
        std::cout << std::setw(shift) << second;

        std::cout << std::setw(0) << std::endl;
    };

    printCompare("out", "left");
    success &= (expectedRight == right);
    for (size_t i = 0; i < gridSize.x; ++i) {

        std::stringstream outStr;
        outStr << "expected: " << expectedOut[i]
               << " got: "     << out[i];
        std::stringstream leftStr;
        leftStr << "expected: " << expectedLeft[i]
                << " got: "     << left[i];

        printCompare(outStr.str(), leftStr.str(), i);

        success &= (expectedOut[i]   == out[i]);
        success &= (expectedLeft[i]  == left[i]);
    }

    if (!success) {
        return { "Invalid output values" };
    }

    return {};
}

}  // namespace Runtime

}  // namespace TestCase
