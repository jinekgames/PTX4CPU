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

#include "logo_ascii.h"


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

bool SaveOutputData(const std::string& filepath, PtxExecArgs& execArgs) {

    std::string output;
    EMULATOR_SerializeArgsJson(execArgs, output);
    
    if (output.empty()) {
        std::cout << "ERROR: Failed to serialize output values" << std::endl;
        return false;
    }

    std::ostream* pOut = &std::cout;
    std::ofstream fout;

    bool ret = true;

    if (filepath.empty()) {
        std::cout << "Execution output:" << std::endl << std::endl;
    } else {
        fout.open(filepath);
        if(fout.is_open()) {
            pOut = &fout;
            std::cout << "Saving execion output to '" << filepath << "'" << std::endl;
        } else {
            std::cout << "ERROR: File '" << filepath << "' opening failed. "
                      << "Execution output will be printed to console:" << std::endl << std::endl;
            ret = false;
        }
    }

    *pOut << output;
    return ret;
}

} // anonimous namespace

// @todo implementation: add ability to export parsed ptx file

int main(size_t argc, char** argv) {
    Parser args(argc, argv);

    if (args.Contains("help")) {

        std::cout <<
R"(This tool traslate PTX assemble file to x86-copatible ones

See usage docs here:

Commands:
   --help      - show this message
   --run       - runs a given kernel
                 Params:
                     --kernel  - name of kernel to execute
                     --args    - path to the .json contaning execution arguments
                     --threads - count of kernel execution threads
                     --save-output  - [optional] path to the .json where execution output will be written
                                      if argument was not specified, output will be printed to console
                                      if argument was specified with empty value, original arguments .json will be used
                 Example:
                     TranslatorCmd.exe --test-run input_file.ptx --kernel _Z9addKernelPiPKiS1_ --args arguments.json --save-output
   --test-load - do a test load and preparsing of a .ptx file
                 Example:
                     TranslatorCmd.exe --test-load input_file.ptx
   --test-json - tests if an arguments configuration .json is valid
                 Example:
                     TranslatorCmd.exe --test-json arguments.json
)" << ASCII_LOGO_CUDA4CPU << std::endl;

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

    } else if (args.Contains("test-json")) {

        std::string argsJsonPath = args["test-json"];
        
        auto pExecVars = ParseArgsJson(ReadFile(argsJsonPath));
        if (!pExecVars) {
            std::cout << "ERROR: Parsing of the execution arguments json failed" << std::endl;
            return 1;
        }

        std::cout << "Json is Valid" << std::endl;
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
        std::string outputJsonPath;
        if (args.contains("save-output")) {
            outputJsonPath = args["save-output"];
            if (outputJsonPath.empty()) {
                outputJsonPath = argsJsonPath;
            }
        }

        std::cout << "Starting execution of kernel '" << kernelName << "'from '"
                  << inputPath << "' in " << threadsCount << " treads" << std::endl;

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
            auto res = SaveOutputData(outputJsonPath, *pExecVars);
            return (res) ? 0 : 1;
        }

        std::cout << "Function execution faled. Error: " << res.msg << std::endl;
        return 1;

    }

        std::cout <<
R"(Invalid command was specified.

Run with '--help' to see available commands
)" << std::endl;

        return 1;
}
