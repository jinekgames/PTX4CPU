#include <cmd_parser/cmd_parser.h>
#include <emulator_api.h>
#include <helpers.h>
#include <utils/string_utils.h>

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>


namespace {

std::string ReadFile(const std::string& filepath) {

    std::ifstream sin(filepath);
    if(!sin.is_open()) {
        std::cout << "ERROR: File '" << filepath << "' opening failed" << std::endl;
        return "";
    }

    std::stringstream input;
    input << sin.rdbuf();
    return input.str();
}

auto CreateTranslator(const std::string& src) {
    PTX4CPU::ITranslator* rawPtr = nullptr;
    EMULATOR_CreateTranslator(&rawPtr, src);

    return std::unique_ptr<PTX4CPU::ITranslator>{rawPtr};
}

auto ParseArgsJson(const std::string& json) {
    auto rawPtr = new PtxExecArgs;
    EMULATOR_ParseArgsJson(rawPtr, json);

    return std::unique_ptr<PtxExecArgs>{rawPtr};
}

} // anonimous namespace

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
                     --kernel  - name of kernel to execute
                     --args    - path to the .json contaning execution arguments
                     --threads - count of kernel execution threads
                 Example:
                     TranslatorCmd.exe --test-run input_file.ptx --kernel _Z9addKernelPiPKiS1_
)" << std::endl;

        return 0;

    } else if (args.Contains("test-load")) {

        std::string inputPath = args["test-load"];

        const auto input = ReadFile(inputPath);
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

        const std::string inputPath = args["test-run"];

        for (const auto& argName : { "kernel", "args", "threads" }) {
            if (!args.Contains(argName)) {
                std::cout << "ERROR: missing --" << argName << " argument" << std::endl;
                return 1;
            }
        }

        const auto kernelName   = args["kernel"];
        const auto argsJsonPath = args["args"];
        if (!IsNumber(args["threads"])) {
            std::cout << "ERROR: Invalid --threads value" << std::endl;
            return 1;
        }
        const auto threadsCount = static_cast<uint32_t>(std::stol(args["threads"]));

        // Parse PTX execution arguments

        auto pExecVars = ParseArgsJson(ReadFile(argsJsonPath));
        if (!pExecVars) {
            std::cout << "ERROR: Parsing of the execution arguments json failed" << std::endl;
            return 1;
        }

        // Create Translator

        const auto input = ReadFile(inputPath);
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

        // Execute PTX

        uint3_32 gridSize = { threadsCount, 1, 1 };

        auto res = pTranslator->ExecuteFunc(kernelName, *pExecVars, gridSize);

        if (res) {
            std::cout << "Kernel finished execution" << std::endl;
            return 0;
        }

        // PTX4CPU::Result res{"end"};

        std::cout << "Function execution faled. Error: " << res.msg << std::endl;
        return 1;

    }

        std::cout <<
R"(No run command was specified.

Run with '--help' to see available commands
)" << std::endl;

        return 1;
}
