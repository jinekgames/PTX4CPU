#pragma once

#include <string>


namespace PTX4CPU {
namespace TransBase {

class Element {

public:

    Element(std::string str) : inputStr(str) {}

    virtual std::string Translate() = 0;

private:

    std::string inputStr;

};

} // namespace TransBase
} // namespace PTX4CPU
