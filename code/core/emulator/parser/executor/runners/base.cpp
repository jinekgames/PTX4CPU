#include "runner.h"


namespace PTX4CPU {
namespace DispatchTable {

Result Branch(ThreadExecutor* pExecutor,
              const Types::Instruction& instruction) {

#ifdef OPT_COMPILE_SAFE_CHECKS
    if (instruction.args.size() != 1) {
        return { "Branch instruction must have an only label argument" };
    }
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    const auto &label = instruction.args.front();

    PRINT_V("Function '%s' jumps to \"%s\"",
            pExecutor->GetFunc()->name.c_str(), label.c_str());
    return pExecutor->Jump(label);
}

Result Return(ThreadExecutor* pExecutor,
              const Types::Instruction& instruction) {

#ifdef OPT_COMPILE_SAFE_CHECKS
    if (instruction.args.size() != 0) {
        return { "Return instruction must not have arguments" };
    }
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    PRINT_V("Function '%s' returned", pExecutor->GetFunc()->name.c_str());
    pExecutor->Finish();
    return {};
}

}  // namespace DispatchTable
}  // namespace PTX4CPU
