#pragma once

#include "element.h"

#include <vector>


namespace PTX4CPU {
namespace TransBase {

class Block : public Element {

public:

    std::string Translate() override;

private:

    std::vector<Element*> elements;

};

} // namespace TransBase
} // namespace PTX4CPU
