#include "string_iterator.h"


namespace StringIteration {

bool IsDelimiter(char symbol, WordDelimiter delimiter) {

    const auto check = [](char c, WordDelimiter base) {
        return GetBaseDelims(base).find(c) != std::string_view::npos;
    };
    for (auto base :
         { Space, NewLine, Punct, Dot, Brackets, MathOperators, BackSlash }) {

        if ((delimiter & base) && check(symbol, base)) {
            return true;
        }
    }
    return false;
}

}  // namespace StringIteration
