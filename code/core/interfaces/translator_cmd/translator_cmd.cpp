#include <cmd_parser/cmd_parser.h>
#include <emulator_api.h>
#include <helpers.h>

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>


std::string OpenPTX(std::string filepath) {
    std::cout << "Using a PTX from: \"" << filepath << "\"" << std::endl;

    std::ifstream sin(filepath);
    if(!sin.is_open()) {
        std::cout << "ERROR: Input file opening failed" << std::endl;
        return "";
    }

    std::stringstream input;
    input << sin.rdbuf();
    return input.str();
}

auto CreateTranslator(const std::string src) {
    PTX4CPU::ITranslator* rawPtr = nullptr;
    EMULATOR_CreateTranslator(&rawPtr, src);

    return std::unique_ptr<PTX4CPU::ITranslator>{rawPtr};
}

struct InputData {
    PTX4CPU::Types::PTXVarList execArgs;
    PTX4CPU::Types::PTXVarList tempVars;
};

// Parces a given json configuring a PTX execution arguments
// @param jsonStr a .json with execution arguments
// @return `InputData` object contating PTX execution arguments and temporary
// variables
InputData ParseJson(const std::string& jsonStr) {


}

template<class T, size_t Size>
std::tuple<PTX4CPU::Types::PTXVarList, std::array<uint64_t, Size>>
MakeExecArgs(std::array<std::vector<T>, Size>& vars, uint32_t count) {

    // @todo implementation: pass args
    // @todo implementation: save result data

    using namespace PTX4CPU;

    for (size_t idx = 0; idx < Size; ++idx) {
        vars[idx].resize(count);
    }

    for (uint32_t i = 0; i < count; ++i) {
        for (size_t idx = 0; idx < Size; ++idx) {
            vars[idx][i] = static_cast<T>(idx);
        }
    }

    // this should be valid for all exec type
    std::array<uint64_t, Size> pVarsConv;
    for (size_t idx = 0; idx < Size; ++idx) {
        pVarsConv[idx] = reinterpret_cast<uint64_t>(vars[idx].data());
    }

    // this is temporary and could be deleted
    uint64_t ppVarsConv[Size];
    for (size_t idx = 0; idx < Size; ++idx) {
        ppVarsConv[idx] = reinterpret_cast<uint64_t>(&pVarsConv[idx]);
    }

    Types::PTXVarList args;
    for (size_t idx = 0; idx < Size; ++idx) {
        args.push_back(
            std::move(
                Types::PTXVarPtr(new Types::PTXVarTyped<Types::PTXType::U64>(
                    &ppVarsConv[idx]
                ))
            )
        );
    }

    return { std::move(args), pVarsConv };
}

// @todo implementation: add ability to export parsed ptx file

int main(size_t argc, char** argv) {
    Parser args(argc, argv);

    if (args.Contains("help")) {

        std::cout <<
R"(This tool traslate PTX assemble file to x86-copatible ones

Commands:
   --help      - show this message
   --test-load - do a test load and preparsing of a .ptx file
                 Example:
                     TranslatorCmd.exe --test-load input_file.ptx
   --test-run  - do a test run of a given kernel with empty arguments
                 Params:
                     --kernel - name of kernel to execute
                 Example:
                     TranslatorCmd.exe --test-run input_file.ptx --kernel _Z9addKernelPiPKiS1_
)" << std::endl;

        return 0;

    } else if (args.Contains("test-load")) {

        std::string inputPath = args["test-load"];

        auto input = OpenPTX(inputPath);
        if (input.empty()) {
            std::cout << "ERROR: Invalid PTX file" << std::endl;
            return 1;
        }

        auto pTranslator{std::move(CreateTranslator(input))};

        if (!pTranslator) {
            std::cout << "ERROR: Failed to create a Translator object" << std::endl;
            return 1;
        }

        std::cout << "Translator was successfully created" << std::endl;
        return 0;

    } else if (args.Contains("test-run")) {

        std::string inputPath = args["test-run"];

        auto input = OpenPTX(inputPath);
        if (input.empty()) {
            std::cout << "ERROR: Invalid PTX file" << std::endl;
            return 1;
        }

        auto pTranslator{std::move(CreateTranslator(input))};

        if (!pTranslator) {
            std::cout << "ERROR: Failed to create a Translator object" << std::endl;
            return 1;
        }

        std::cout << "Translator was successfully created" << std::endl;

        auto kernelName = args["kernel"];

        const uint32_t threadsCount = 3;
        const size_t varsCount = 3;
        std::array<std::vector<int32_t>, varsCount> vars;
        auto [execArgs, ppVarsConv] = MakeExecArgs(vars, threadsCount);
        uint3_32 gridSize = { threadsCount, 1, 1 };

        auto res = pTranslator->ExecuteFunc(kernelName, execArgs, gridSize);

        if (res) {
            std::cout << "Kernel finished execution" << std::endl;
            return 0;
        }

        std::cout << "Function execution faled. Error: " << res.msg << std::endl;
        return 1;

    }

        std::cout <<
R"(No run command was specified.

Run with '--help' to see available commands
)" << std::endl;

        return 1;
}
