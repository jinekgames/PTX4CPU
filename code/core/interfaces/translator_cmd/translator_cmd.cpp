#include <cmd_parser.h>
#include <emulator_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

int main(size_t argc, char** argv) {
    Parser args(argc, argv);

    if (args.Contains("help")) {
        std::cout <<
R"(This tool traslate PTX assemble file to x86-copatible ones

Commands:
   --help      - show this message
   --test-load - do a test load of a .ptx file
                 Example:  TranslatorCmd.exe --test-load input_file.ptx
)";
        return 0;
    } else if (args.Contains("test-load")) {
        std::string inputPath = args["test-load"];

        std::ifstream sin(inputPath);
        if(!sin.is_open()) {
            std::cout << "ERROR: Input file opening failed";
            return 1;
        }

        std::stringstream input;
        input << sin.rdbuf();

        std::unique_ptr<PTX2ASM::ITranslator> pTranslator;

        {
            PTX2ASM::ITranslator* rawPtr = nullptr;
            EMULATOR_CreateTranslator(&rawPtr, input.str());

            decltype(pTranslator) swapPtr{rawPtr};
            pTranslator.swap(swapPtr);
        }

        if (!pTranslator) {
            std::cout << "ERROR: Failed to create a Translator object";
            return 1;
        }

        std::cout << "Translator was successfully created";
        return 0;
    }

        std::cout <<
R"(No run command was specified.

Run with '--help' to see available commands
)";
        return 1;
}