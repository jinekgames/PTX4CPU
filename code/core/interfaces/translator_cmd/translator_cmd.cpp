#include <cmd_parser.h>
#include <emulator_api.h>

#include <fstream>
#include <iostream>
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
    } else if (args.Contains("test-load")) {
        std::string inputPath = args["test-load"];

        std::ifstream sin(inputPath);
        if(!sin.is_open()) {
            std::cout << "ERROR: Input file opening failed";
            return 1;
        }

        std::stringstream input;
        input << sin.rdbuf();

        PTX2ASM::ITranslator* translator = nullptr;
        EMULATOR_CreateTranslator(&translator, input.str());
        if (!translator) {
            std::cout << "ERROR: Failed to create Translator object";
            return 1;
        }

        std::cout << "Translator was successfully created";
        return 0;

        delete translator;
        translator = nullptr;
    }



    // else if() {
        // if (!args.Contains("output")) {
        //     std::cout << "ERROR: Incorrect arguement. \"convert\" command must contain \"--output\" argument. "
        //                  "See \"--help\"";
        //     return 1;
        // }
        // std::string inputPath = args["convert"];
        // std::string outputPath = args["output"];

        // std::ifstream sin(inputPath);
        // if(!sin.is_open()) {
        //     std::cout << "ERROR: Input file opening failed";
        // }
        // std::ofstream sout(outputPath);
        // if(!sout.is_open()) {
        //     std::cout << "ERROR: Output file opening failed";
        // }

        // std::stringstream input;
        // input << sin.rdbuf();

        // PTX2ASM::ITranslator* translator = nullptr;
        // EMULATOR_CreateTranslator(&translator, input.str());
        // if (!translator) {
        //     std::cout << "ERROR: Failed to create Translator object";
        //     return 1;
        // }

        // if (!translator->Translate()) {
        //     std::cout << "ERROR: Transation failed";
        // }
        // sout << translator->GetResult();

        // delete translator;
        // translator = nullptr;
    // }

    return 0;
}