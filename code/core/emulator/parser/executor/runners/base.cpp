#include "runner.h"


using namespace PTX4CPU;

Result DispatchTable::Return(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter&) {
    auto& iter = pExecutor->GetIter();
    iter.Shift(pExecutor->GetFunc().end - iter.Offset());
    return {};
}
