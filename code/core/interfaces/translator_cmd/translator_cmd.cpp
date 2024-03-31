#include <cmd_parser.h>
#include <emulator_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>


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

        auto res = pTranslator->ExecuteFunc(kernelName);

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
