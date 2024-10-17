#include <instruction_runner.h>
#include <logger/logger.h>
#include <string_utils.h>


namespace PTX4CPU {

std::unordered_map<std::string,
    InstructionRunner::RunnerFunc> InstructionRunner::m_DispatchTable = {};

// Runners' regitration template

class RunnerRegistrator {
public:
    RunnerRegistrator(std::string command, InstructionRunner::RunnerFunc func);
};

RunnerRegistrator::RunnerRegistrator(std::string command, InstructionRunner::RunnerFunc func) {
    InstructionRunner::m_DispatchTable.emplace(command, func);
}

#define RegisterRunner(command, runnerFuncName)                             \
    Result runnerFuncName(ThreadExecutor* pExecutor,                        \
                          const Types::Instruction& instruction);           \
    MapRunner(command, runnerFuncName)

#define MapRunner(command, runnerFuncName)                                  \
    RunnerRegistrator CONCAT(reg, __LINE__){command, runnerFuncName}


// Registring of all the runners and their internals

namespace DispatchTable {

// runners/base.cpp

RegisterRunner("ret", Return);


// runners/memory.cpp

/**
 * Allocates given `count` of vars with given `name`
 * Returns count of allocated vars
*/
template<Types::PTXType ptxType>
static uint64_t RegisterMemoryInternal(ThreadExecutor* pExecutor,
                                       const std::string& name,
                                       const uint64_t count);
RegisterRunner(".reg", RegisterMemory);

/**
 * Dereference var with name `ptrName` into the var named `storeName`
*/
template<Types::PTXType ptxType>
static Result LoadParamInternal(ThreadExecutor* pExecutor,
                                const std::string& valueName,
                                const std::string& ptrName);
RegisterRunner("ld.param", LoadParam);
MapRunner("ld.global", LoadParam);

/**
 * Dereference var with name `ptrName` and store there variable from the var
 * named `storeName`
*/
template<Types::PTXType ptxType>
static Result SetParamInternal(ThreadExecutor* pExecutor,
                               const std::string& valueName,
                               const std::string& ptrName);
RegisterRunner("st.global", SetParam);

template<bool copyAsReference>
static Result CopyVarInternal(ThreadExecutor* pExecutor,
                              const Types::Instruction& instruction);
RegisterRunner("cvta.to.global", CopyVarAsReference);
RegisterRunner("mov",            CopyVarAsValue);


// runners/memory.cpp

RegisterRunner("mul.hi",   MulHi);
RegisterRunner("mul.lo",   MulLo);
RegisterRunner("mul.wide", MulWide);

RegisterRunner("add", Add);

}  // namespace DispatchTable

#undef RegisterRunner
#undef MapRunner

}  // namespace PTX4CPU
