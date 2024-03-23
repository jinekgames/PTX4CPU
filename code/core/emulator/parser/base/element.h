#pragma once

#include <string>


namespace PTX2ASM {
namespace TransBase {

class Element {

public:

    Element(std::string str) : inputStr(str) {}

    virtual std::string Translate() = 0;

private:

    std::string inputStr;

};

} // namespace TransBase
} // namespace PTX2ASM
