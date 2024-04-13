#include "runner.h"

#include <parser.h>
#include <string_utils.h>

#include <charconv>

#ifdef COMPILE_SAFE_CHECKS
#ifdef WIN32
#include <Windows.h>
#endif
#endif

using namespace PTX4CPU;

template<Types::PTXType ptxType>
uint64_t DispatchTable::RegisterMemoryInternal(const ThreadExecutor* pExecutor,
                                               const std::string& name, const uint64_t count) {

    size_t i;
    for (i = 1; i <= count; ++i) {
        const std::string numberedName = name + std::to_string(i);
        pExecutor->m_pVarsTable->AppendVar<ptxType>(numberedName);
    }
    return i - 1;
}

Result DispatchTable::RegisterMemory(const ThreadExecutor* pExecutor,
                                     InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // .reg   .b32   %r<5>;

    // move back to start for parsing
    iter.Reset();
    const auto [nameWithCount, desc] = Parser::ParsePtxVar(iter.GetString());

    // Parse real name and count
    StringIteration::SmartIterator nameIter{nameWithCount};

    const auto name = nameIter.ReadWord2();

    const auto countStr = nameIter.ReadWord2();
    uint64_t count = 0;
    std::from_chars(countStr.data(), countStr.data() + countStr.length(), count);

    uint64_t allocatedCount = 0;
    PTXTypedOp(desc.type,
        allocatedCount = RegisterMemoryInternal<_Runtime_Type_>(pExecutor, name, count);
    )

    if (allocatedCount != count)
        return { "Failed to allocate " + std::to_string(count - allocatedCount) + " variables" };
    return {};
}

namespace {
template<Types::PTXType ptxType>
Result Dereference(Types::PTXVar* ptrVar, Types::getVarType<ptxType>& value) {

    using realType = std::remove_cvref_t<decltype(value)>;

    const auto ptrPtxType = Types::PTXType::
#ifdef SYSTEM_ARCH_64
        S64;
#else
        S32;
#endif

    auto ptr = ptrVar->Get<ptrPtxType>();

#ifdef COMPILE_SAFE_CHECKS
    if (!ptr) {
        return { "Null pointer dereference" };
    }
#ifdef WIN32
    if (IsBadReadPtr(reinterpret_cast<void*>(ptr), static_cast<UINT_PTR>(sizeof(realType)))) {
        return { "Invalid pointer. Possible reading access violation" };
    }
#endif
#endif

    value = *reinterpret_cast<realType*>(ptr);

    return {};
}
}

template<Types::PTXType ptxType>
Result DispatchTable::LoadParamInternal(const ThreadExecutor* pExecutor,
                                        const std::string& valueName,
                                        const std::string& ptrName) {

    auto valueNameParsed = Parser::ParseVectorName(valueName);
    auto ptrNameParsed   = Parser::ParseVectorName(ptrName);

    auto ptrVar = pExecutor->m_pVarsTable->FindVar(ptrNameParsed.name);
#ifdef COMPILE_SAFE_CHECKS
    if (!ptrVar) {
        return { "Variable \"" + ptrName + "\" storing the pointer is undefined" };
    }
#endif

    Types::getVarType<ptxType> value;
    auto result = Dereference<ptxType>(ptrVar, value);

    if (!result) {
        PRINT_E(result.msg.c_str());
        return { "Dereferece of " + ptrName + " failed" };
    }

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->m_pVarsTable->FindVar(valueName)) {
        return { "Variable \"" + valueName + "\" storing the dereferanced value is undefined" };
    }
#endif
    pExecutor->m_pVarsTable->GetValue<ptxType>(valueName) = value;

    return {};
}

Result DispatchTable::LoadParam(const ThreadExecutor* pExecutor,
                                InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // ld.param.u64   %rd1,   [_Z9addKernelPiPKiS1__param_0];

    const auto typeStr = iter.ReadWord2();
    const auto type = Types::GetFromStr(typeStr);

    const auto valueName = iter.ReadWord2();
    const auto ptrName = iter.ReadWord2();

    PTXTypedOp(type,
        return LoadParamInternal<_Runtime_Type_>(pExecutor, valueName, ptrName);
    )

    return { "Failed to found pointer dereference runner for the given type: " + typeStr };
}

template<bool copyAsReference>
Result DispatchTable::CopyVar(const ThreadExecutor* pExecutor,
                                      InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // cvta.to.global.u64   %rd5,   %rd3;

    const auto typeStr = iter.ReadWord2();
    const auto type = Types::GetFromStr(typeStr);

    const auto dstName = iter.ReadWord2();
    const auto srcName = iter.ReadWord2();

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->m_pVarsTable->FindVar(dstName)) {
        return { "Variable \"" + dstName + "\" storing the destination value is undefined" };
    }
    if (!pExecutor->m_pVarsTable->FindVar(srcName)) {
        return { "Variable \"" + srcName + "\" storing the source value is undefined" };
    }
    // Original virtual variable will be overriden, so we additionally check if types are copatible
    if (pExecutor->m_pVarsTable->GetVar(dstName).GetPTXType() !=
        pExecutor->m_pVarsTable->GetVar(srcName).GetPTXType()) {
        return { "Types of source \"" + srcName + "\" and destination \"" + dstName + "\" convert/copy variables are not copatible" };
    }
#endif

    decltype(auto) srcVar = pExecutor->m_pVarsTable->GetVar(srcName);
    pExecutor->m_pVarsTable->DeleteVar(dstName);
    if constexpr (copyAsReference) {
        pExecutor->m_pVarsTable->AppendVar(dstName, srcVar.MakeReference());
    } else {
        pExecutor->m_pVarsTable->AppendVar(dstName, srcVar.MakeCopy());
    }

    return {};
}

Result DispatchTable::CopyVarAsReference(const ThreadExecutor* pExecutor,
                                         InstructionRunner::InstructionIter& iter) {
    return CopyVar<true>(pExecutor, iter);
}

Result DispatchTable::CopyVarAsValue(const ThreadExecutor* pExecutor,
                                     InstructionRunner::InstructionIter& iter) {
    return CopyVar<false>(pExecutor, iter);
}


