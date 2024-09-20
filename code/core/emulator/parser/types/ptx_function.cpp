#include "ptx_function.h"


namespace PTX4CPU {
namespace Types {

Function::Function(Function&& right) {

    Move(*this, right);
}

Function& Function::operator = (Function&& right) {

    Move(*this, right);
    return *this;
}

void Function::Move(Function& left, Function& right) {

    if (&left == &right)
        return;

    left.name       = std::move(right.name);
    left.attributes = std::move(right.attributes);
    left.arguments  = std::move(right.arguments);
    left.returns    = std::move(right.returns);
    left.start      = right.start;
    left.end        = right.end;

    right.start = Data::Iterator::Npos;
    right.end   = Data::Iterator::Npos;
}

}  // namespace Types
}  // namespace PTX4CPU
