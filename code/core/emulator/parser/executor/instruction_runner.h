#pragma once

#include <functional>
#include <memory>
#include <string>

#include <parser_types.h>
#include <string_utils.h>
#include <utils/base_types.h>
#include <utils/result.h>



namespace PTX4CPU {

class RunnerRegistrator;

class ThreadExecutor;

class InstructionRunner {

private:

    InstructionRunner() = delete;

public:

    using RunnerFuncRet   = Result;
    using InstructionIter = const StringIteration::SmartIterator<const std::string>;
    using RunnerFunc      = std::function<RunnerFuncRet(ThreadExecutor*,
                                                        const Types::Instruction&)>;

    InstructionRunner(const Types::Instruction& instruction, ThreadExecutor* pExecutor);
    InstructionRunner(const InstructionRunner&) = delete;
    InstructionRunner(InstructionRunner&&)      = delete;
    ~InstructionRunner()                        = default;

    InstructionRunner& operator = (const InstructionRunner&) = delete;
    InstructionRunner& operator = (InstructionRunner&&)      = delete;

private:

    void FindRunner();

public:

    Result Run();

private:

    const Types::Instruction& m_Instruction;

    ThreadExecutor* m_pExecutor;

    RunnerFunc m_Runner;

    static std::unordered_map<std::string, RunnerFunc> m_DispatchTable;

    friend class RunnerRegistrator;

};

}  // namespace PTX4CPU
