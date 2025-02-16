#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

#include "ptx_types.h"
#include "parser_data.h"
#include "virtual_var.h"


namespace PTX4CPU {
namespace Types {

struct Instruction {

    explicit Instruction(const std::string& intructionStr);

    struct Predicate {

        explicit Predicate(const std::string& predicateStr);

        // Represents whether the positive or negative result is target
        bool isNegative = false;
        // Variable name storing the processed parameter
        std::string varName;

        static constexpr auto PRED_PREFIX_SYMB   = '@';
        static constexpr auto PRED_NEGATIVE_SYMB = '!';
    };

    std::string    GetStrType() const;
    Types::PTXType GetPtxType() const;

    // Predicative execution param (optional)
    std::optional<Predicate> predicate;
    // Instruction name
    std::string name;

    using ArgsList = std::vector<std::string>;

    // Instruction execution arguments
    ArgsList args;
};

/**
 * PTX function's description with the list of instructions
*/
struct Function
{
public:

    Function()                 = default;
    Function(const Function&)  = delete;
    Function(Function&& right) = default;
    ~Function()                = default;

    Function& operator = (const Function&)  = delete;
    Function& operator = (Function&& right) = default;

public:

    // Insert isntructions from current position till the end of the block
    void InsertInstructions(Data::Iterator& iter);

public:

    // A name of the function stated in the PTX file
    std::string name;

    // function attribute to it's optional value
    using Attributes   = std::unordered_map<std::string, std::string>;
    // name of argument with argument description
    using ArgWithName  = std::pair<std::string, PtxVarDesc>;
    // argument name to it's type
    using Arguments    = std::vector<ArgWithName>;
    // returning value name to it's type
    using Returns      = std::vector<ArgWithName>;
    // List of function's instructions
    using Instructions = std::vector<Instruction>;

    Attributes   attributes;
    Arguments    arguments;
    Returns      returns;
    Instructions instructions;

};

using FuncsList = std::vector<Types::Function>;

}  // namespace Types
}  // namespace PTX4CPU
