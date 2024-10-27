#include "utils.h"

#include <fstream>
#include <sstream>


std::string ReadFile(const std::string& filepath) {

    std::ifstream sin(filepath);
    if (!sin.is_open()) {
        return "";
    }

    std::stringstream input;
    input << sin.rdbuf();
    return input.str();
}
