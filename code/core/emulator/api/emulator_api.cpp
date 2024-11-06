#include <emulator_api.h>

#include <emulator/emulator.h>
#include <ext_parsers/ext_parsers.h>
#include <logger/logger.h>
#include <parser/parser.h>


extern "C" {

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateEmulator(PTX4CPU::IEmulator** ppEmulator,
                        const std::string& sourceCode) {

    *ppEmulator = new PTX4CPU::Emulator(sourceCode);
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_ParseArgsJson(PtxExecArgs* pInputData, const std::string& jsonStr) {

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
EMULATOR_ProcessArgs(const PtxFuncDescriptor pKernel,
                     const void* const* ppArgs,
                     PtxExecArgs* pInputData) {

    if(!pInputData) {
        PRINT_E("Null Input data object passed");
        return;
    }

    *pInputData = nullptr;

    if (!pKernel) {
        PRINT_E("Null Kernel descriptor passed");
        return;
    }

    if(!PTX4CPU::Parser::IsKernelFunction(*pKernel)) {
        PRINT_E("Non kernel descriptor passed");
        return;
    }

    const auto res = ParseCudaArgs(ppArgs, pKernel->arguments, *pInputData);

    if(!res) {
        PRINT_E("Failed to parse runtime CUDA args: %s", res.msg.c_str());
    }
}

EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_SerializeArgsJson(const PtxExecArgs& inputData, std::string& jsonStr) {

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
