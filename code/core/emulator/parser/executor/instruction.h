#pragma once

#include <functional>
#include <memory>
#include <string>

#include <parser_types.h>
#include <result.h>
#include <utils/base_types.h>



namespace PTX4CPU {

class RunnerRegistrator;

class ThreadExecutor;

class InstructionRunner {

private:

    InstructionRunner() = default;

public:

    using RunnerFuncArg = const ThreadExecutor*;
    using RunnerFuncRet = Result;
    using RunnerFunc    = std::function<RunnerFuncRet(RunnerFuncArg)>;

    InstructionRunner(const std::string& instruction, const ThreadExecutor* pExecutor);
    InstructionRunner(const InstructionRunner&) = delete;
    InstructionRunner(InstructionRunner&&) = delete;
    ~InstructionRunner() = default;

    InstructionRunner& operator = (const InstructionRunner&) = delete;
    InstructionRunner& operator = (InstructionRunner&&) = delete;

private:

    void FindRunner();

public:

    Result Run();

private:

    std::string m_Instruction;

    const ThreadExecutor* m_pExecutor;

    RunnerFunc m_Runner;

    static std::unordered_map<std::string, RunnerFunc> m_DispatchTable;

    friend class RunnerRegistrator;

};

}  // namespace PTX4CPU
