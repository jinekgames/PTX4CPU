#include <instruction.h>

#include <executor.h>
#include <helpers.h>
#include <logger/logger.h>


using namespace PTX4CPU;

InstructionRunner::InstructionRunner(const std::string& instruction, const ThreadExecutor* pExecutor)
    : m_Instruction{instruction}
    , m_InstructionIter{m_Instruction}
    , m_pExecutor{pExecutor} {

    if (m_pExecutor) {
        PRINT_V("[%d,%d,%d]:%lu: > %s",
                m_pExecutor->m_ThreadId.x, m_pExecutor->m_ThreadId.y, m_pExecutor->m_ThreadId.z,
                m_pExecutor->m_DataIter.GetOffset(),
                instruction.c_str());
    }

    FindRunner();
}

void InstructionRunner::FindRunner() {

    /**
     * A command could be
     * .reg
     * or
     * ld.param.u64
     * Erase type from the command, it will be processed by the runner
    */

    auto command = m_InstructionIter.ReadWord2();

    auto dotIdx = command.find_last_of('.');
    if (dotIdx != 0 && dotIdx != std::string::npos)
    {
        command.erase(dotIdx);
        // shift back to type start to read from the runner
        m_InstructionIter.Shift(dotIdx - m_InstructionIter.GetOffset());
    }

    auto found = m_DispatchTable.find(command);
    if (found != m_DispatchTable.end())
        m_Runner = found->second;
}

Result InstructionRunner::Run() {

    if (!m_pExecutor)
        return {"Invalid executor"};

    if (m_Runner)
        return m_Runner(m_pExecutor, m_InstructionIter);

    return {Result::Code::NotOk, "Runner for the given instruction not found. Skipped (" +
                                 m_InstructionIter.GetString() + ")"};
}
