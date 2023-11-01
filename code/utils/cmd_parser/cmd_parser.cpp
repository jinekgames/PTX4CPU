#include <cmd_parser.h>

#include <string_view>
#include <vector>

void Parser::Parse(size_t argc, char* argv[]) {
    std::vector<std::string_view> args(argc);
    for (size_t i = 1; i < argc; ++i) { // skip first arg with executable path
        args[i - 1] = argv[i];
    }

    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg = args[i];
        if (arg._Starts_with("-")) {
            size_t firstLetterIdx = 0;
            while (arg[firstLetterIdx] == '-') {
                ++firstLetterIdx;
            }
            emplace(&arg[firstLetterIdx], args[i + 1]);
        }
    }
}
