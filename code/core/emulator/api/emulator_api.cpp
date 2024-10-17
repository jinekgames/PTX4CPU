#include <emulator_api.h>

#include <json_parser/parser.h>
#include <logger/logger.h>
#include <emulator/emulator.h>


extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateEmulator(PTX4CPU::IEmulator** ppEmulator,
                        const std::string& sourceCode) {

    *ppEmulator = new PTX4CPU::Emulator(sourceCode);
}

extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_ParseArgsJson(PtxExecArgs* inputData, const std::string& jsonStr) {

    PtxInputData retData;
    auto res = PTX4CPU::ParseJson(retData, jsonStr);

    if (!res) {
        PRINT_E(res.msg.c_str());
        *inputData = nullptr;
        return;
    }

    *inputData = new PtxInputData{std::move(retData)};
}

extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
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
