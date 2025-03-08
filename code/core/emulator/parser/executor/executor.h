#pragma once

#include <parser_types.h>
#include <utils/base_types.h>
#include <utils/result.h>

#include <utility>
#include <vector>

namespace PTX4CPU {

class ThreadExecutor final {

public:

    using InstructionIndexType = Types::Function::IndexType;

private:

    ThreadExecutor() = default;

public:

    ThreadExecutor(const Types::Function* pFunc,
                   const std::shared_ptr<Types::VarsTable>& pArguments,
                   const BaseTypes::uint3_32& threadId);
    ThreadExecutor(const ThreadExecutor&)  = delete;
    ThreadExecutor(ThreadExecutor&& right) = default;
    ~ThreadExecutor()                      = default;

    ThreadExecutor& operator = (const ThreadExecutor&)  = delete;
    ThreadExecutor& operator = (ThreadExecutor&& right) = default;

    void Reset() const;

    /**
     * Moves iteration to the end of instruction list
    */
    void Finish() const;

    /**
     * Run a given count of instruction from the last break point
    */
    Result Run(InstructionIndexType instructionsCount);

    /**
     * Run a function till the end
    */
    Result Run() {
        return Run(m_pFunc->instructions.size());
    }

    /**
     * @brief Jumps before a given offset from the function beginning
     */
    Result Jump(InstructionIndexType offset);

    /**
     * @brief Jumps before a given label
     */
    Result Jump(const std::string& label);

    /**
     * @brief
     * Retrieve the virtual variable for the given name
     * If a temp value was passed insted of existed name, it will be created as
     * a temp virtual variable.
     * @param type operation type (needed for creating a temp var)
     * @param arg  argument string
     * @return virtual variable pointers
    */
    Types::ArgumentPair RetrieveArg(
        Types::PTXType type, const std::string& arg) const;
    /**
     * @brief
     * Retrieve the virtual variables from the list
     * @note
     * See `ThreadExecutor::RetrieveArg` description
    */
    std::vector<Types::ArgumentPair> RetrieveArgs(
        Types::PTXType type, Types::Instruction::ArgsList args) const;

    inline auto             GetTID()   const { return m_ThreadId; }
    inline auto             GetPos()   const { return m_InstructionPosition; }
    Types::VarsTable*       GetTable()       { return m_pVarsTable.get(); }
    const Types::VarsTable* GetTable() const { return m_pVarsTable.get(); }
    const Types::Function*  GetFunc()  const { return m_pFunc; }

    void DebugLogVars() const;

private:

    bool CheckPredicate(const Types::Instruction::Predicate& predicate) const;

private:

    /**
     * Inserts predefined constants to the vars table
     * (e.g. `%tid`)
    */
    void AppendConstants() const;

    BaseTypes::uint4_32 m_ThreadId;

    const Types::Function* m_pFunc;

    // Position of instrunction in the given fucntion.
    mutable InstructionIndexType m_InstructionPosition = 0;

    // Stored for keeping an ownershit
    const std::shared_ptr<Types::VarsTable> m_pArguments;

    std::shared_ptr<Types::VarsTable> m_pVarsTable;

};

}  // namespace PTX4CPU
