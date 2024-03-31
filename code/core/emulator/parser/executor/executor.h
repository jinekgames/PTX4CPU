#pragma once

#include <result.h>
#include <parser_data.h>
#include <parser_types.h>
#include <utils/base_types.h>


namespace PTX4CPU {

class ThreadExecutor {

private:

    ThreadExecutor() = default;

public:

    ThreadExecutor(const Data::Iterator& iterator, const Types::Function& func,
                   std::shared_ptr<Types::VarsTable> arguments, const int3& threadId);
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
    Result Run(Data::Iterator::Size instructionsCount) const;

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

    mutable std::shared_ptr<Types::VarsTable> m_VarsTable;

    friend class InstructionRunner;
    friend class DispatchTable;

};

}  // namespace PTX4CPU
