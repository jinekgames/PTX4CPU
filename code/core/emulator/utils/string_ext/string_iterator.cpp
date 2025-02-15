#include "string_iterator.h"


namespace StringIteration {

bool IsDelimiter(char symbol, WordDelimiter delimiter) {

    if (GetBaseDelims(delimiter).find(symbol) != std::string::npos) {
        return true;
    }
    return false;
}

}  // namespace StringIteration
