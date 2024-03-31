#include <instruction.h>
#include <dispatch_table.h>

#include <executor.h>

namespace PTX4CPU {
class DispatchTable {
public:

    static Result Return(const ThreadExecutor* pExecutor) {
        auto& iter = pExecutor->m_DataIter;
        iter.Shift(pExecutor->m_Func.end - iter.GetOffset());
        return {};
    }

    RegisterRunner("ret", Return);

};  // class DispatchTable

DispatchTable dispatchTable;
}  // namespace PTX4CPU
