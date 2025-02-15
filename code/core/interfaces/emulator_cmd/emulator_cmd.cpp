#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include <cmd_parser.h>
#include <emulator_api.h>
#include <helpers.h>
#include <utils/string_utils.h>

#include "logo_ascii.h"


namespace {

std::string ReadFile(const std::string& filepath) {

    std::ifstream sin(filepath);
    if (!sin.is_open()) {
        std::cout << "ERROR: File '" << filepath << "' opening failed" << std::endl;
        return "";
    }

    std::stringstream input;
    input << sin.rdbuf();
    return input.str();
}

auto CreateEmulator(const std::string& src) {
    PTX4CPU::IEmulator* rawPtr = nullptr;
    EMULATOR_CreateEmulator(&rawPtr, src);

    return rawPtr;
}

auto ParseArgsJson(const std::string& json) {
    PTX4CPU::PtxExecArgs rawPtr = PTX4CPU_NULL_HANDLE;
    EMULATOR_CreateArgsJson(&rawPtr, json);

    return rawPtr;
}

bool SaveOutputData(const std::string& filepath,
                    PTX4CPU::PtxExecArgs& execArgs) {

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
        if (fout.is_open()) {
            pOut = &fout;
            std::cout << "Saving execution output to '" << filepath << "'" << std::endl;
        } else {
            std::cout << "ERROR: File '" << filepath << "' opening failed. "
                      << "Execution output will be printed to console:" << std::endl << std::endl;
            ret = false;
        }
    }

    *pOut << output;
    return ret;
}

void CleanUp(PTX4CPU::PtxExecArgs args, PTX4CPU::IEmulator* pEmulator)
{
    if(args != PTX4CPU_NULL_HANDLE) {
        EMULATOR_DestroyArgs(args);
    }
    if(pEmulator != nullptr) {
        EMULATOR_DestroyEmulator(pEmulator);
    }
}

} // anonimous namespace

// @todo implementation: add ability to export preprocessed ptx file

int main(int argc, char** argv) {
    Parser args(argc, argv);

    if (args.Contains("help")) {

        std::cout <<
R"(
See usage docs here: https://github.com/jinekgames/PTX4CPU/blob/main/README.md

Commands:
   --help         - show this message
   --run          - run a given kernel
                    Params:
                        --kernel  - name of kernel to execute
                        --args    - path to the .json contaning execution arguments
                        --threads - count of kernel execution threads
                        --save-output  - [optional] path to the .json where execution output will be written
                                         if argument was not specified, output will be printed to console
                                         if argument was specified with empty value, original arguments .json will be used
                    Example:
                        EmulatorCmd.exe --test-run input_file.ptx --kernel _Z9addKernelPiPKiS1_ --args arguments.json --save-output
   --test-load    - do a test load and preparsing of a .ptx file
                    Example:
                        EmulatorCmd.exe --test-load input_file.ptx
   --test-json    - test if an arguments configuration .json is valid
                    Example:
                        EmulatorCmd.exe --test-json arguments.json
   -v, --version,
   --about        - show "about" info
)" << std::endl;

        return 0;

    } else if (args.Contains("v") || args.Contains("version") || args.Contains("about")) {

        std::cout << "PTX4CPU v" << PTX4CPU_VERSION << std::endl;
        std::cout << "This tool allows to run PTX assemble files on the CPU power" << std::endl;
        std::cout << ASCII_LOGO_CUDA4CPU << std::endl;
        return 0;

    } else if (args.Contains("test-load")) {

        std::string inputPath = args["test-load"];

        const auto input = ReadFile(inputPath);
        if (input.empty()) {
            std::cout << "ERROR: Invalid PTX file" << std::endl;
            return 1;
        }

        auto pEmulator = CreateEmulator(input);

        if (pEmulator == PTX4CPU_NULL_HANDLE) {
            std::cout << "ERROR: Failed to create a Emulator object" << std::endl;
            return 1;
        }

        std::cout << "Emulator was successfully created" << std::endl;

        CleanUp(PTX4CPU_NULL_HANDLE, pEmulator);

        return 0;

    } else if (args.Contains("test-json")) {

        std::string argsJsonPath = args["test-json"];

        auto execVars = ParseArgsJson(ReadFile(argsJsonPath));
        if (execVars == PTX4CPU_NULL_HANDLE) {
            std::cout << "ERROR: Parsing of the execution arguments json failed" << std::endl;
            return 1;
        }

        std::cout << "Json is Valid" << std::endl;

        CleanUp(execVars, nullptr);

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

        std::cout << "Starting execution of kernel '" << kernelName
                  << "'from '" << inputPath << "' in " << threadsCount
                  << " treads" << std::endl;

        // Parse PTX execution arguments

        std::cout << "Using argument from '" << argsJsonPath << "'"
                  << std::endl;

        auto execVars = ParseArgsJson(ReadFile(argsJsonPath));
        if (execVars == PTX4CPU_NULL_HANDLE) {
            std::cout << "ERROR: Parsing of the execution arguments json failed" << std::endl;
            return 1;
        }

        // Create Emulator

        const auto input = ReadFile(inputPath);
        if (input.empty()) {
            std::cout << "ERROR: Invalid PTX file" << std::endl;
            return 1;
        }

        auto pEmulator = CreateEmulator(input);

        if (pEmulator == PTX4CPU_NULL_HANDLE) {
            std::cout << "ERROR: Failed to create an Emulator object" << std::endl;
            return 1;
        }

        std::cout << "Emulator was successfully created" << std::endl;

        // Execute PTX

        BaseTypes::uint3_32 gridSize = { threadsCount, 1, 1 };

        auto res = pEmulator->ExecuteFunc(kernelName, execVars, gridSize);

        if (res) {
            std::cout << "Kernel finished execution" << std::endl;
            auto res = SaveOutputData(outputJsonPath, execVars);
            return (res) ? 0 : 1;
        }

        CleanUp(execVars, pEmulator);

        std::cout << "Function execution faled. Error: " << res.msg << std::endl;
        return 1;

    }

    std::cout <<
R"(Invalid command was specified.

Run with '--help' to see available commands
)" << std::endl;

    return 1;
}
