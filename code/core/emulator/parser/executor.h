#pragma once

#include <result.h>
#include <parser_data.h>
#include <parser_types.h>
#include <utils/base_types.h>


namespace PTX2ASM {

// @todo refactoring: move realizations in a special file
class ThreadExecutor {

public:

    ThreadExecutor(const Data::Iterator& iterator, const Types::Function& func,
                   std::shared_ptr<Types::VarsTable> arguments, const int3& threadId)
        : m_ThreadId{threadId}
        , m_DataIter{iterator}
        , m_Func{func}
        , m_Arguments{arguments}
        , m_VarsTable(m_Arguments.get()) {

        Reset();
    }
    ThreadExecutor(const ThreadExecutor&) = delete;
    ThreadExecutor(ThreadExecutor&& right)
        : m_DataIter{std::move(right.m_DataIter)}
        , m_Func{std::move(right.m_Func)}
        , m_VarsTable{std::move(right.m_VarsTable)}
        , m_ThreadId{std::move(right.m_ThreadId)} {}
    ~ThreadExecutor() = default;

    ThreadExecutor& operator = (const ThreadExecutor&) = delete;
    ThreadExecutor& operator = (ThreadExecutor&& right) {
        if(this == &right)
            return *this;

        m_DataIter  = std::move(right.m_DataIter);
        m_Func      = std::move(right.m_Func);
        m_VarsTable = std::move(right.m_VarsTable);
        m_ThreadId  = std::move(right.m_ThreadId);

        return *this;
    }

    // Prepare for running
    void Reset() const {
        m_DataIter.Reset();
        m_DataIter.Shift(m_Func.start);
        m_VarsTable.Clear();
    }

    /**
     * Run a given count of instruction in a thread from the last break point
    */
    Result Run(Data::Iterator::Size instructionsCount) const {
        PRINT_I("ThreadExecutor[%llu,%llu,%llu]: Starting a function execution (offset:%llu)",
                m_ThreadId.x, m_ThreadId.y, m_ThreadId.z, m_DataIter.GetOffset());

        for (; m_DataIter.IsValid() && m_DataIter.GetOffset() - m_Func.start < instructionsCount;
             m_DataIter.Next()) {

            decltype(auto) instStr = m_DataIter.ReadInstruction();

            PRINT_V("ThreadExecutor[%llu,%llu,%llu]: > %s",
                    m_ThreadId.x, m_ThreadId.y, m_ThreadId.z, instStr.c_str());
            // @todo imlementation: do instructions executors
        }

        if (m_DataIter.GetOffset() < m_Func.end)
            PRINT_I("ThreadExecutor[%llu,%llu,%llu]: Execution paused (offset:%llu)",
                    m_ThreadId.x, m_ThreadId.y, m_ThreadId.z, m_DataIter.GetOffset());
        else
            PRINT_I("ThreadExecutor[%llu,%llu,%llu]: Execution finished",
                    m_ThreadId.x, m_ThreadId.y, m_ThreadId.z);

        return {};
    }

    /**
     * Run a function till the end
    */
    Result Run() const {
        return Run(m_Func.end - m_Func.start);
    }

    auto GetTID() const { return m_ThreadId; }

private:

    int3 m_ThreadId;

    Data::Iterator m_DataIter;

    Types::Function m_Func;

    std::shared_ptr<Types::VarsTable> m_Arguments;

    mutable Types::VarsTable m_VarsTable;

};

}  // namespace PTX2ASM
