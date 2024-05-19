#pragma once

#include <instruction.h>
#include <string_utils.h>


namespace PTX4CPU {

// Runners' regitration template

class RunnerRegistrator {
public:
    RunnerRegistrator(std::string command, InstructionRunner::RunnerFunc func);
};

#define RegisterRunner(command, runnerFuncName)                                                                 \
    static Result runnerFuncName(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter);    \
    RunnerRegistrator CONCAT(reg, __LINE__){command, runnerFuncName}

#define RegisterRunnerFuncless(command, runnerFuncName)                                                         \
    RunnerRegistrator CONCAT(reg, __LINE__){command, runnerFuncName}


// Registring of all the runners and their internals

class DispatchTable {
public:

    // runners/base.cpp

    RegisterRunner("ret", Return);


    // runners/memory.cpp

    /**
     * Allocates given `count` of vars with given `name`
     * Returns count of allocated vars
    */
    template<Types::PTXType ptxType>
    static uint64_t RegisterMemoryInternal(const ThreadExecutor* pExecutor,
                                           const std::string& name, const uint64_t count);
    RegisterRunner(".reg", RegisterMemory);

    /**
     * Dereference var with name `ptrName` into the var named `storeName`
    */
    template<Types::PTXType ptxType>
    static Result LoadParamInternal(const ThreadExecutor* pExecutor,
                                    const std::string& valueName,
                                    const std::string& ptrName);
    RegisterRunner("ld.param", LoadParam);
    RegisterRunnerFuncless("ld.global", LoadParam);

    /**
     * Dereference var with name `ptrName` and store there variable from the var
     * named `storeName`
    */
    template<Types::PTXType ptxType>
    static Result SetParamInternal(const ThreadExecutor* pExecutor,
                                   const std::string& valueName,
                                   const std::string& ptrName);
    RegisterRunner("st.global", SetParam);

    template<bool copyAsReference>
    static Result CopyVarInternal(const ThreadExecutor* pExecutor,
                                  InstructionRunner::InstructionIter& iter);
    RegisterRunner("cvta.to.global", CopyVarAsReference);
    RegisterRunner("mov",            CopyVarAsValue);


    // runners/memory.cpp

    template<Types::PTXType type>
    static Types::PTXVarPtr CreateTempValueVarTyped(const std::string& value);
    static Types::PTXVarPtr CreateTempValueVar(Types::PTXType type, const std::string& value);

    using pOpProc = Result(*)(Types::PTXType, Types::PTXVar&, Types::PTXVar&, Types::PTXVar&,
                          char, char, char);

    static Result Op2Internal(const ThreadExecutor* pExecutor,
                              InstructionRunner::InstructionIter& iter,
                              pOpProc opProc);
    RegisterRunner("mul.hi",   MulHi);
    RegisterRunner("mul.lo",   MulLo);
    RegisterRunner("mul.wide", MulWide);

    RegisterRunner("add", Add);

};  // class DispatchTable

#undef RegisterRunner
#undef RegisterRunnerFuncless

}  // namespace PTX4CPU
