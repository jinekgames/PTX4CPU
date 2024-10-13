#include "runner.h"


namespace PTX4CPU {
namespace DispatchTable {

Result Return(ThreadExecutor* pExecutor,
              const Types::Instruction& instruction) {

    PRINT_V("Function '%s' returned", pExecutor->GetFunc()->name.c_str());
    pExecutor->Finish();
    return {};
}

}  // namespace DispatchTable
}  // namespace PTX4CPU
