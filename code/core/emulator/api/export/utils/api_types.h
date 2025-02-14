#pragma once


#define PTX4CPU_NULL_HANDLE nullptr


namespace PTX4CPU {
namespace Types {

struct PtxInputData;

struct Function;

}  // namespace Types

using PtxExecArgs       = PTX4CPU::Types::PtxInputData*;

using PtxFuncDescriptor = PTX4CPU::Types::Function*;

}  // namespace PTX4CPU

