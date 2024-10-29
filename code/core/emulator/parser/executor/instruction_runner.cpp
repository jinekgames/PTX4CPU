#include <instruction_runner.h>

#include <format>

#include <executor.h>
#include <helpers.h>
#include <logger/logger.h>
#include <utils/string_utils.h>


using namespace PTX4CPU;

InstructionRunner::InstructionRunner(const Types::Instruction& instruction,
                                     ThreadExecutor* pExecutor)
    : m_Instruction{instruction}
    , m_pExecutor{pExecutor} {

    if (m_pExecutor) {
        PRINT_V("[%d,%d,%d]:%llu > %s %s",
                m_pExecutor->GetTID().x,
                m_pExecutor->GetTID().y,
                m_pExecutor->GetTID().z,
                m_pExecutor->GetPos(),
                m_Instruction.name.c_str(),
                Merge(std::string(", "), m_Instruction.args).c_str());
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

    auto command = m_Instruction.name;

    auto dotIdx = command.find_last_of('.');
    if (dotIdx != 0 && dotIdx != std::string::npos) {
        command.erase(dotIdx);
    }

    auto found = m_DispatchTable.find(command);
    if (found != m_DispatchTable.end()) {
        m_Runner = found->second;
    }
}

Result InstructionRunner::Run() {

    if (!m_pExecutor) {
        return {"Invalid executor"};
    }

    if (m_Runner) {
        return m_Runner(m_pExecutor, m_Instruction);
    }

    return {Result::Code::NotOk,
            FormatString("Runner for the given instruction not found. "
                         "Skipped `{} {}`",
                         m_Instruction.name,
                         Merge(std::string(", "), m_Instruction.args))};
}
