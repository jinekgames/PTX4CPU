#pragma once

#include <result.h>
#include <parser_types.h>
#include <utils/base_types.h>


namespace PTX4CPU {

class ThreadExecutor {

private:

    ThreadExecutor() = default;

public:

    ThreadExecutor(const Data::Iterator& iterator, const Types::Function& func,
                   const std::shared_ptr<Types::VarsTable>& arguments,
                   const BaseTypes::uint3_32& threadId);
    ThreadExecutor(const ThreadExecutor&) = delete;
    ThreadExecutor(ThreadExecutor&& right);
    ~ThreadExecutor() = default;

    ThreadExecutor& operator = (const ThreadExecutor&) = delete;
    ThreadExecutor& operator = (ThreadExecutor&& right);

    // Prepare for running
    void Reset() const;

    /**
     * Run a given count of instruction in a thread from the last break point
    */
    Result Run(Data::Iterator::SizeType instructionsCount) const;

    /**
     * Run a function till the end
    */
    Result Run() const {
        return Run(m_Func.end - m_Func.start);
    }

    auto                   GetTID()   const { return m_ThreadId; }
    Types::VarsTable*      GetTable() const { return m_pVarsTable.get(); }
    const Types::Function& GetFunc()  const { return m_Func; }
    Data::Iterator&        GetIter()  const { return m_DataIter; }

private:

    void AppendConstants() const;

    BaseTypes::uint4_32 m_ThreadId;

    mutable Data::Iterator m_DataIter;

    Types::Function m_Func;

    // Stored for keeping an ownershit
    const std::shared_ptr<Types::VarsTable> m_pArguments;

    mutable std::shared_ptr<Types::VarsTable> m_pVarsTable;

    friend class InstructionRunner;

};

}  // namespace PTX4CPU
