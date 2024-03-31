#include <instruction.h>

#include <executor.h>
#include <logger.h>
#include <string_utils.h>


namespace PTX4CPU {

InstructionRunner::InstructionRunner(const std::string& instruction, const ThreadExecutor* pExecutor)
    : m_Instruction{instruction}
    , m_pExecutor{pExecutor} {

    if (pExecutor)
        PRINT_V("[%d,%d,%d]:%lu: > %s",
                pExecutor->m_ThreadId.x, pExecutor->m_ThreadId.y, pExecutor->m_ThreadId.z,
                pExecutor->m_DataIter.GetOffset(),
                instruction.c_str());

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

    const StringIteration::SmartIterator iter{m_Instruction};
    auto command = iter.ReadWord2();

    auto dotIdx = command.find_last_of('.');
    if (dotIdx != 0 && dotIdx != std::string::npos)
        command.erase(dotIdx);

    auto found = m_DispatchTable.find(command);
    if (found != m_DispatchTable.end())
        m_Runner = found->second;
}

Result InstructionRunner::Run() {

    if (!m_pExecutor)
        return {"Invalid executor"};

    if (m_Runner)
        return m_Runner(m_pExecutor);

    return {Result::Code::NotOk, "Runner for the given instruction not found. Skipped"};
}

}  // namespace PTX4CPU
