#include <executor.h>

#include <format>

#include <instruction.h>


namespace PTX4CPU {

ThreadExecutor::ThreadExecutor(const Data::Iterator& iterator, const Types::Function& func,
                               const std::shared_ptr<Types::VarsTable>& arguments, const int3& threadId)
    : m_ThreadId{threadId}
    , m_DataIter{iterator}
    , m_Func{func}
    , m_Arguments{arguments} {

    m_VarsTable = std::make_shared<Types::VarsTable>(m_Arguments.get());
    Reset();
}

ThreadExecutor::ThreadExecutor(ThreadExecutor&& right)
    : m_ThreadId{std::move(right.m_ThreadId)}
    , m_DataIter{std::move(right.m_DataIter)}
    , m_Func{std::move(right.m_Func)}
    , m_Arguments{std::move(right.m_Arguments)}
    , m_VarsTable{std::move(right.m_VarsTable)} {}

ThreadExecutor& ThreadExecutor::operator = (ThreadExecutor&& right) {

    if(this == &right)
        return *this;

    m_DataIter  = std::move(right.m_DataIter);
    m_Func      = std::move(right.m_Func);
    m_VarsTable = std::move(right.m_VarsTable);
    m_ThreadId  = std::move(right.m_ThreadId);

    return *this;
}

void ThreadExecutor::Reset() const {
    m_DataIter.Reset();
    m_DataIter.Shift(m_Func.start);
    if (m_VarsTable)
        m_VarsTable->Clear();
}

Result ThreadExecutor::Run(Data::Iterator::Size instructionsCount) const {

    const std::string logPrefix = std::vformat("ThreadExecutor[{},{},{}]",
        std::make_format_args(m_ThreadId.x, m_ThreadId.y, m_ThreadId.z));

    PRINT_I("%s: Starting a function execution (offset:%llu)",
            logPrefix.c_str(), m_DataIter.GetOffset());

    for (; m_DataIter.IsValid() && m_DataIter.GetOffset() - m_Func.start < instructionsCount;
         m_DataIter.Next()) {

        decltype(auto) instStr = m_DataIter.ReadInstruction();

        InstructionRunner runner{instStr, this};
        auto res = runner.Run();

        if(!res) {
            if (res == Result::Code::NotOk) {
                PRINT_W("%s Execution warning (offset:%llu): %s",
                        logPrefix.c_str(), m_DataIter.GetOffset(), res.msg.c_str());
            } else {
                res.msg = std::vformat("(offset:{}): {}",
                    std::make_format_args(m_DataIter.GetOffset(), res.msg));
                return res;
            }
        }
    }

    PRINT_I("%s: Execution paused (offset:%llu)",
            logPrefix.c_str(), m_DataIter.GetOffset());

    return {};
}

}  // namespace PTX4CPU
