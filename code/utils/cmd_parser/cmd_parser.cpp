#include "cmd_parser.h"

#include <string_view>
#include <vector>

namespace {

bool IsCommand(const std::string_view& arg) {
    return arg._Starts_with("-");
}

} // anonimous namespace

void Parser::Parse(size_t argc, char* argv[]) {
    std::vector<std::string_view> args(argc);
    for (size_t i = 1; i < argc; ++i) { // skip first arg with executable path
        args[i - 1] = argv[i];
    }

    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg = args[i];
        if (IsCommand(arg)) {
            size_t firstLetterIdx = 0;
            while (arg[firstLetterIdx] == '-') {
                ++firstLetterIdx;
            }
            auto value = IsCommand(args[i + 1]) ? "" : args[i + 1];
            emplace(&arg[firstLetterIdx], value);
        }
    }
}
