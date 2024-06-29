#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

#include "ptx_types.h"
#include "parser_data.h"
#include "virtual_var.h"


namespace PTX4CPU {
namespace Types {

struct Function
{
    Function()                = default;
    Function(const Function&) = default;
    Function(Function&& right);
    ~Function()               = default;
    Function& operator = (const Function&) = delete;
    Function& operator = (Function&& right);

private:

    static void Move(Function& left, Function& right);

public:

    // A name of the function stated in the PTX file
    std::string name;
    // function attribute to it's optional value
    std::unordered_map<std::string, std::string> attributes;
    // argument name to it's type
    std::unordered_map<std::string, PtxVarDesc> arguments;
    // returning value name to it's type
    std::unordered_map<std::string, PtxVarDesc> returns;
    // Index of m_Data pointed to the first instruction of the function body
    Data::Iterator::SizeType start = Data::Iterator::Npos;
    // Index of m_Data pointed to the first index after the last instruction of the function body
    Data::Iterator::SizeType end   = Data::Iterator::Npos;
};

using FuncsList = std::vector<Types::Function>;

}  // namespace Types
}  // namespace PTX4CPU
