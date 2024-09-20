#include "string_iterator.h"


namespace StringIteration {

bool IsDelimiter(char symbol, WordDelimiter delimiter) {

    for (const auto& delimData : baseDelimsTable) {
        if (delimData.first & delimiter &&
            delimData.second.find(symbol) != std::string::npos) {
                return true;
        }
    }
    return false;
}

}  // namespace StringIteration
