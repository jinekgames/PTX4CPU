#include "utils.h"

#include <fstream>
#include <sstream>


std::string ReadFile(const std::string& filepath) {

    std::string ret;

    std::ifstream sin(filepath);
    if (!sin.is_open()) {
        return "";
    }

    std::stringstream input;
    input << sin.rdbuf();

    sin.close();

    ret = input.str();
    return ret;
}
