#pragma once

#include "element.h"
#include "variable.h"

#include <vector>


namespace PTX2ASM {
namespace TransBase {

class Instruction : public Element {

public:

    std::string Translate() override;

private:

    std::vector<Variable*> operands;

};

} // namespace TransBase
} // namespace PTX2ASM
