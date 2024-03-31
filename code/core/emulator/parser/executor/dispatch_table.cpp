#include <instruction.h>
#include <dispatch_table.h>

#include <logger.h>


namespace PTX4CPU {

std::unordered_map<std::string,
    InstructionRunner::RunnerFunc> InstructionRunner::m_DispatchTable = {};

RunnerRegistrator::RunnerRegistrator(std::string command, InstructionRunner::RunnerFunc func) {
    InstructionRunner::m_DispatchTable.emplace(command, func);
}

}  // namespace PTX4CPU
