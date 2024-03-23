#pragma once

#include "element.h"

#include "utils.h"


namespace PTX2ASM {
namespace TransBase {

struct Variable : public Element {

public:

    enum Use {
        InstructOperand,
        FuncArguement,
        Definition
    };

    Variable(std::string str, Use useType) : use(useType), Element(str) {}

    std::string Translate() override {
        std::string out;
        switch (use)
        {
        case InstructOperand:
            size_t begin = NextLiteral(inputStr, 0);
            size_t end = NextLiteral(inputStr, begin);
            out = inputStr.substr(begin, end - begin);
            out.erase(); // erase prefix % symbol
            break;

        case FuncArguement
        
        default:
            break;
        }
    }

private:

    Use use;

};

} // namespace TransBase
} // namespace PTX2ASM