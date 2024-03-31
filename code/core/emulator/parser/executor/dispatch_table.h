#pragma once

#include <instruction.h>
#include <string_utils.h>


namespace PTX4CPU {

class RunnerRegistrator {
public:
    RunnerRegistrator(std::string command, InstructionRunner::RunnerFunc func);
};

#define RegisterRunner(command, runnerFuncName)                                            \
    RunnerRegistrator CONCAT(reg, __LINE__){command, runnerFuncName}
// InstructionRunner::RunnerFuncRet runnerFuncName(InstructionRunner::RunnerFuncArg);

}  // namespace PTX4CPU
