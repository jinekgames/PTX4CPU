#pragma once


namespace PTX4CPU {
namespace Types {

struct PtxInputData;

struct Function;

}  // namespace Types
}  // namespace PTX4CPU


using PtxExecArgs       = PTX4CPU::Types::PtxInputData*;

using PtxFuncDescriptor = PTX4CPU::Types::Function*;
