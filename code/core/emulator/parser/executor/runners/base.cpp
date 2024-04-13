#include "runner.h"


using namespace PTX4CPU;

Result DispatchTable::Return(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter&) {
    auto& iter = pExecutor->m_DataIter;
    iter.Shift(pExecutor->m_Func.end - iter.GetOffset());
    return {};
}
