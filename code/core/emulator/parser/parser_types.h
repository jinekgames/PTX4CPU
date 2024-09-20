#pragma once

#include <algorithm>
#include <array>
#include <map>
#include <stdfloat>
#include <string>
#include <unordered_map>
#include <utility>

#include "types/parser_data.h"
#include "types/ptx_function.h"
#include "types/ptx_types.h"
#include "types/virtual_var.h"


namespace PTX4CPU {
namespace Types {

struct PtxProperties {
    std::pair<int8_t, int8_t> version = { 0, 0 };
    int32_t target      = 0;
    int32_t addressSize = 0;

    bool IsValid() {
        return (version.first || version.second) &&
                version.first >= 0 && version.second >= 0 &&
                target > 0 &&
                addressSize > 0;
    }
};

}  // namespace Types
}  // namespace PTX4CPU


struct PtxInputData {
    PTX4CPU::Types::PTXVarList execArgs;
    PTX4CPU::Types::PTXVarList outVars;
    PTX4CPU::Types::PTXVarList tempVars;
};
