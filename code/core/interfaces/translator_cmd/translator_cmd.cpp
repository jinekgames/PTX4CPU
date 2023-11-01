#include <cmd_parser.h>
#include <emulator_api.h>

#include <fstream>
#include <iostream>
#include <sstream>

int main(size_t argc, char** argv) {
    Parser args(argc, argv);
    
    if (args.Contains("help")) {
        std::cout <<
            "This tool traslate PTX assemble file to x86-copatible ones\n"
            "Commands:\n"
            "   --help    - show this message\n"
            "   --convert - do translation\n"
            "               specify input file right after the command, then output file with \"--output\" argument\n"
            "               Example:  TranslatorCmd.exe --convert input_file.ptx --output output_file.asm\n";
    }
    else if (args.Contains("convert")) {
        if (!args.Contains("output")) {
            std::cout << "ERROR: Incorrect arguement. \"convert\" command must contain \"--output\" argument. "
                         "See \"--help\"";
            return 1;
        }
        std::string inputPath = args["convert"];
        std::string outputPath = args["output"];

        std::ifstream sin(inputPath);
        if(!sin.is_open()) {
            std::cout << "ERROR: Input file opening failed";
        }
        std::ofstream sout(outputPath);
        if(!sout.is_open()) {
            std::cout << "ERROR: Output file opening failed";
        }

        std::stringstream input;
        input << sin.rdbuf();

        PTX2ASM::ITranslator* translator = nullptr;
        EMULATOR_CreateTranslator(&translator, input.str());
        if (!translator) {
            std::cout << "ERROR: Failed to create Translator object";
            return 1;
        }

        if (!translator->Translate()) {
            std::cout << "ERROR: Transation failed";
        }
        sout << translator->GetResult();

        delete translator;
        translator = nullptr;
    }

    return 0;
}