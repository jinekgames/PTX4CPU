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

RegisterRunner("bra", Branch);

RegisterRunner("ret", Return);


// runners/memory.cpp

RegisterRunner(".reg", RegisterMemory);

RegisterRunner("cvta.to.global", CopyVarAsReference);

RegisterRunner("ld.global", LoadParam);
MapRunner("ld.param", LoadParam);

RegisterRunner("mov", CopyVarAsValue);

RegisterRunner("st.global", SetParam);


// runners/math.cpp

RegisterRunner("add", Add);
RegisterRunner("and", And);
RegisterRunner("shl", Shl);
RegisterRunner("sub", Sub);
RegisterRunner("div", Div);

RegisterRunner("mul.hi",   MulHi);
RegisterRunner("mul.lo",   MulLo);
RegisterRunner("mul.wide", MulWide);

RegisterRunner("mad.hi",   MaddHi);
RegisterRunner("mad.lo",   MaddLo);
RegisterRunner("mad.wide", MaddWide);

RegisterRunner("setp.ge", LogicalGE);
RegisterRunner("setp.eq", LogicalEQ);
RegisterRunner("setp.lt", LogicalLT);
RegisterRunner("setp.ne", LogicalNE);

RegisterRunner("fma.rn", FmaRn);

}  // namespace DispatchTable

#undef RegisterRunner
#undef MapRunner

}  // namespace PTX4CPU
