#include <executor.h>

#include <format>

#include <instruction_runner.h>


using namespace PTX4CPU;

ThreadExecutor::ThreadExecutor(const Data::Iterator& iterator, const Types::Function& func,
                               const std::shared_ptr<Types::VarsTable>& arguments,
                               const BaseTypes::uint3_32& threadId)
    : m_ThreadId{threadId}
    , m_DataIter{iterator}
    , m_Func{func}
    , m_pArguments{arguments} {

    m_pVarsTable = std::make_shared<Types::VarsTable>(m_pArguments.get());
    Reset();
}

ThreadExecutor::ThreadExecutor(ThreadExecutor&& right)
    : m_ThreadId{std::move(right.m_ThreadId)}
    , m_DataIter{std::move(right.m_DataIter)}
    , m_Func{std::move(right.m_Func)}
    , m_pArguments{std::move(right.m_pArguments)}
    , m_pVarsTable{std::move(right.m_pVarsTable)} {}

ThreadExecutor& ThreadExecutor::operator = (ThreadExecutor&& right) {

    if(this == &right)
        return *this;

    m_DataIter   = std::move(right.m_DataIter);
    m_Func       = std::move(right.m_Func);
    m_pVarsTable = std::move(right.m_pVarsTable);
    m_ThreadId   = std::move(right.m_ThreadId);

    return *this;
}

void ThreadExecutor::Reset() const {
    m_DataIter.Reset();
    m_DataIter.Shift(m_Func.start);
    if (m_pVarsTable) {
        m_pVarsTable->Clear();
        AppendConstants();
    }
}

Result ThreadExecutor::Run(Data::Iterator::SizeType instructionsCount) const {

    const std::string logPrefix = std::vformat("ThreadExecutor[{},{},{}]",
        std::make_format_args(m_ThreadId.x, m_ThreadId.y, m_ThreadId.z));

    PRINT_I("%s: Starting a function execution (offset:%llu)",
            logPrefix.c_str(), m_DataIter.Offset());

    for (; m_DataIter.IsValid() && m_DataIter.Offset() - m_Func.start < instructionsCount;
         ++m_DataIter) {

        decltype(auto) instStr = m_DataIter.ReadInstruction();

        InstructionRunner runner{instStr, this};
        auto res = runner.Run();

        if(!res) {
            if (res == Result::Code::NotOk) {
                PRINT_W("%s Execution warning (offset:%llu): %s",
                        logPrefix.c_str(), m_DataIter.Offset(), res.msg.c_str());
            } else {
                res.msg = std::vformat("(offset:{}): {}",
                    std::make_format_args(m_DataIter.Offset(), res.msg));
                return res;
            }
        }
    }

    PRINT_I("%s: Execution paused (offset:%llu)",
            logPrefix.c_str(), m_DataIter.Offset());

    return {};
}

void ThreadExecutor::AppendConstants() const {

    m_pVarsTable->AppendVar<Types::PTXType::U32, 4>("%tid", &m_ThreadId.x);
}
