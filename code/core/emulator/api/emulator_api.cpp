#include <emulator_api.h>

#include <arg_parsers/arg_parsers.h>
#include <emulator/emulator.h>
#include <logger/logger.h>
#include <parser/parser.h>


extern "C" {

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateEmulator(
    PTX4CPU::IEmulator** ppEmulator,
    const std::string&   sourceCode) {

    *ppEmulator = new PTX4CPU::Emulator(sourceCode);
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_DestroyEmulator(
    PTX4CPU::IEmulator* pEmulator) {

    delete pEmulator;
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateArgs(
    PTX4CPU::PtxExecArgs*            pInputData,
    const PTX4CPU::PtxFuncDescriptor kernel,
    const void* const*               ppArgs) {

    if(!pInputData) {
        PRINT_E("Null Input data object passed");
        return;
    }

    *pInputData = nullptr;

    if (!kernel) {
        PRINT_E("Null Kernel descriptor passed");
        return;
    }

    if(!PTX4CPU::Parser::IsKernelFunction(*kernel)) {
        PRINT_E("Non kernel descriptor passed");
        return;
    }

    const auto res =
        PTX4CPU::ParseCudaArgs(ppArgs, kernel->arguments, pInputData);

    if(!res) {
        PRINT_E("Failed to parse runtime CUDA args: %s", res.msg.c_str());
    }
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateArgsJson(
    PTX4CPU::PtxExecArgs* pInputData,
    const std::string&    jsonStr) {

    if(!pInputData) {
        PRINT_E("Null Input data object passed");
        return;
    }

    PTX4CPU::Types::PtxInputData retData;
    auto res = PTX4CPU::ParseJson(retData, jsonStr);

    if (!res) {
        PRINT_E(res.msg.c_str());
        *pInputData = nullptr;
        return;
    }

    *pInputData = new PTX4CPU::Types::PtxInputData{std::move(retData)};
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_DestroyArgs(
    PTX4CPU::PtxExecArgs inputData) {

    delete inputData;
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_SerializeArgsJson(
    PTX4CPU::PtxExecArgs inputData,
    std::string&         jsonStr) {

    std::string retStr;
    auto res = PTX4CPU::ExtractJson(*inputData, retStr);
    if (!res) {
        PRINT_E("Failed to parse arguments json. Error: %s", res.msg.c_str());
        jsonStr.clear();
        return;
    }

    jsonStr = std::move(retStr);
}

}  // extern "C"
