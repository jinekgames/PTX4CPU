#include <executor.h>

#include <format>

#include <instruction_runner.h>
#include <parser.h>


using namespace PTX4CPU;

ThreadExecutor::ThreadExecutor(const Types::Function* pFunc,
                               const std::shared_ptr<Types::VarsTable>& pArguments,
                               const BaseTypes::uint3_32& threadId)
    : m_ThreadId{threadId}
    , m_pFunc{pFunc}
    , m_pArguments{pArguments} {

    m_pVarsTable = std::make_shared<Types::VarsTable>(m_pArguments.get());
    Reset();
}

void ThreadExecutor::Reset() const {

    m_InstructionPosition = 0;
    if (m_pVarsTable) {
        m_pVarsTable->Clear();
        AppendConstants();
    }
}

void ThreadExecutor::Finish() const {
    m_InstructionPosition = m_pFunc->instructions.size();
}

Result ThreadExecutor::Run(Data::Iterator::SizeType instructionsCount) {

    const std::string logPrefix = FormatString("ThreadExecutor[{},{},{}]",
        m_ThreadId.x, m_ThreadId.y, m_ThreadId.z);

    PRINT_I("%s: Starting a function '%s' execution (offset:%llu of %llu)",
            logPrefix.c_str(), m_pFunc->name.c_str(),
            m_InstructionPosition, m_pFunc->instructions.size());

    DebugLogVars();

    Data::Iterator::SizeType runIdx = 0;

    for (; m_InstructionPosition < m_pFunc->instructions.size() &&
           runIdx < instructionsCount;
         ++m_InstructionPosition, ++runIdx) {

        decltype(auto) instruction =
            m_pFunc->instructions[m_InstructionPosition];

        InstructionRunner runner{instruction, this};
        auto res = runner.Run();

        if (!res) {
            if (res != Result::Code::Fail) {
                PRINT_W("%s Execution warning (offset:%llu): %s",
                        logPrefix.c_str(), m_InstructionPosition,
                        res.msg.c_str());
            } else {
                res.msg = FormatString("(offset:{}): {}",
                                       m_InstructionPosition, res.msg);
                return res;
            }
        }
    }

    if(m_InstructionPosition > m_pFunc->instructions.size()) {
        --m_InstructionPosition;
    }

    PRINT_I("%s: Execution paused (offset:%llu of %llu)", logPrefix.c_str(),
            m_InstructionPosition, m_pFunc->instructions.size());

    return {};
}

Types::ArgumentPair ThreadExecutor::RetrieveArg(
    Types::PTXType type, const std::string& arg) const {

    Types::ArgumentPair ret;

    std::string argName = arg;

    const bool isDereference = Parser::ExtractDereference(argName);

    if(argName.contains('+')) {
        PRINT_E("Inline offset shift is not supported");
    }

    const auto desc  = Parser::ParseVectorName(argName);
    Types::PTXVarPtr pVar{m_pVarsTable->FindVar(desc.name)};

    constexpr auto VAR_NAME_PREFIX = '%';
    if (!pVar) {
        // number constant
        if (argName.front() != VAR_NAME_PREFIX) {
            Types::PTXVarPtr pTempVar;
            PTXTypedOp(type,
                pTempVar = Types::CreateTempValueVarTyped<_PtxType_>(argName);
            )
            pVar.swap(pTempVar);
        } else {
            PRINT_E("Failed to retrieve argument \"%s\"", argName.c_str());
        }
    }
    // pointer dereference
    else if (isDereference) {
        constexpr auto systemPtrType = Types::GetSystemPtrType();
        auto ptr = reinterpret_cast<void *>(
            pVar->Get<systemPtrType>());

        Types::PTXVarPtr pTempVar;
        PTXTypedOp(type,
            pTempVar = Types::CreateTempVarFromPointerTyped<_PtxType_>(ptr);
        )
        pVar.swap(pTempVar);
    }

    return { std::move(pVar), desc.key };
}

std::vector<Types::ArgumentPair> ThreadExecutor::RetrieveArgs(
    Types::PTXType type, Types::Instruction::ArgsList args) const {

    std::vector<Types::ArgumentPair> ret;
    ret.reserve(args.size());
#ifdef OPT_EXTENDED_VARIABLES_LOGGING
    PRINT_V("Instruction args:");
#endif
    for (const auto& arg : args) {
        ret.push_back(RetrieveArg(type, arg));
#ifdef OPT_EXTENDED_VARIABLES_LOGGING
        PRINT_V("%s : %s", arg.c_str(), std::to_string(*ret.back().first).c_str());
#endif
    }
    return ret;
}

void ThreadExecutor::DebugLogVars() const {
#ifdef OPT_EXTENDED_VARIABLES_LOGGING
    PRINT_V("Executor arguments\n%s", std::to_string(*m_pVarsTable).c_str());
#endif
}

void ThreadExecutor::AppendConstants() const {

    m_pVarsTable->AppendVar<Types::PTXType::U32, 4>("%tid", &m_ThreadId.x);
}
