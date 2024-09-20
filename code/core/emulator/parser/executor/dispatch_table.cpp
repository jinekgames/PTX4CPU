#include <instruction_runner.h>
#include <dispatch_table.h>

#include <logger/logger.h>


using namespace PTX4CPU;

std::unordered_map<std::string,
    InstructionRunner::RunnerFunc> InstructionRunner::m_DispatchTable = {};

RunnerRegistrator::RunnerRegistrator(std::string command, InstructionRunner::RunnerFunc func) {
    InstructionRunner::m_DispatchTable.emplace(command, func);
}

DispatchTable dispatchTable;
