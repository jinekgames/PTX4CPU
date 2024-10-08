#include "runner.h"

#include <parser.h>
#include <string_utils.h>

#include <charconv>

#ifdef COMPILE_SAFE_CHECKS
#ifdef WIN32
#include <Windows.h>
#endif  // #ifdef WIN32
#endif  // #ifdef COMPILE_SAFE_CHECKS

using namespace PTX4CPU;

template<Types::PTXType ptxType>
uint64_t DispatchTable::RegisterMemoryInternal(const ThreadExecutor* pExecutor,
                                               const std::string& name, const uint64_t count) {

    size_t i;
    for (i = 1; i <= count; ++i) {
        const std::string numberedName = name + std::to_string(i);
        pExecutor->GetTable()->AppendVar<ptxType>(numberedName);
    }
    return i - 1;
}

Result DispatchTable::RegisterMemory(const ThreadExecutor* pExecutor,
                                     InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // .reg   .b32   %r<5>;

    // move back to start for parsing
    iter.Reset();
    const auto [nameWithCount, desc] = Parser::ParsePtxVar(iter.Data());

    // Parse real name and count
    StringIteration::SmartIterator nameIter{nameWithCount};

    const auto name = nameIter.ReadWord();

    const auto countStr = nameIter.ReadWord();
    uint64_t count = 0;
    std::from_chars(countStr.data(), countStr.data() + countStr.length(), count);

    uint64_t allocatedCount = 0;
    PTXTypedOp(desc.type,
        allocatedCount = RegisterMemoryInternal<_PtxType_>(pExecutor, name, count);
    )

    if (allocatedCount != count)
        return { "Failed to allocate " + std::to_string(count - allocatedCount) + " variables" };
    return {};
}

namespace {

template<Types::PTXType ptxType>
Result Dereference(Types::PTXVar* ptrVar, Types::getVarType<ptxType>& value, char ptrKey = 'x') {

    using realType = std::remove_cvref_t<decltype(value)>;

    const auto ptrPtxType = Types::PTXType::
#ifdef SYSTEM_ARCH_64
        S64;
#else
        S32;
#endif

    auto ptr = ptrVar->Get<ptrPtxType>(ptrKey);

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

template<Types::PTXType ptxType>
Result DereferenceAndSet(Types::PTXVar* ptrVar, const Types::getVarType<ptxType>& value, char ptrKey = 'x') {

    using realType = std::remove_cvref_t<decltype(value)>;

    constexpr auto PtrType = Types::GetSystemPtrType();

    auto ptr = reinterpret_cast<realType*>(ptrVar->Get<PtrType>(ptrKey));

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

    *ptr = value;

    return {};
}

}  // anonimous namespace

template<Types::PTXType ptxType>
Result DispatchTable::LoadParamInternal(const ThreadExecutor* pExecutor,
                                        const std::string& valueFullName,
                                        const std::string& ptrFullName) {

    const auto valueDesc = Parser::ParseVectorName(valueFullName);
    const auto ptrDesc   = Parser::ParseVectorName(ptrFullName);

    auto ptrVar = pExecutor->GetTable()->FindVar(ptrDesc.name);
#ifdef COMPILE_SAFE_CHECKS
    if (!ptrVar) {
        return { "Variable \"" + ptrFullName + "\" storing the pointer is undefined" };
    }
#endif

    Types::getVarType<ptxType> value;
    auto result = Dereference<ptxType>(ptrVar, value, ptrDesc.key);

    if (!result) {
        PRINT_E(result.msg.c_str());
        return { "Loading of param " + ptrFullName + " failed" };
    }

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->GetTable()->FindVar(valueDesc.name)) {
        return { "Variable \"" + valueFullName + "\" storing the dereferanced value is undefined" };
    }
#endif
    pExecutor->GetTable()->GetValue<ptxType>(valueDesc.name, valueDesc.key) = value;

    return {};
}

Result DispatchTable::LoadParam(const ThreadExecutor* pExecutor,
                                InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // ld.param.u64   %rd1,   [_Z9addKernelPiPKiS1__param_0];

    const auto typeStr = iter.ReadWord();
    const auto type    = Types::StrToPTXType(typeStr);

    const auto valueFullName = iter.ReadWord();
    const auto ptrFullName   = iter.ReadWord();

    Result result;
    PTXTypedOp(type,
        result = LoadParamInternal<_PtxType_>(pExecutor, valueFullName, ptrFullName);
    )

    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Dereference of " + ptrFullName + " failed" };
    }
    return {};
}

template<Types::PTXType ptxType>
Result DispatchTable::SetParamInternal(const ThreadExecutor* pExecutor,
                                       const std::string& valueFullName,
                                       const std::string& ptrFullName) {

    const auto valueDesc = Parser::ParseVectorName(valueFullName);
    const auto ptrDesc   = Parser::ParseVectorName(ptrFullName);

    auto ptrVar = pExecutor->GetTable()->FindVar(ptrDesc.name);
#ifdef COMPILE_SAFE_CHECKS
    if (!ptrVar) {
        return { "Variable \"" + ptrFullName + "\" storing the pointer is undefined" };
    }
#endif

    auto value = pExecutor->GetTable()->GetValue<ptxType>(valueDesc.name, valueDesc.key);
    auto result = DereferenceAndSet<ptxType>(ptrVar, value, ptrDesc.key);

    if (!result) {
        PRINT_E(result.msg.c_str());
        return { "Loading of param " + ptrFullName + " failed" };
    }

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->GetTable()->FindVar(valueDesc.name)) {
        return { "Variable \"" + valueFullName + "\" storing the dereferanced value is undefined" };
    }
#endif

    return {};
}

Result DispatchTable::SetParam(const ThreadExecutor* pExecutor,
                                InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // ld.param.u64   %rd1,   [_Z9addKernelPiPKiS1__param_0];

    const auto typeStr = iter.ReadWord();
    const auto type    = Types::StrToPTXType(typeStr);

    const auto ptrFullName   = iter.ReadWord();
    const auto valueFullName = iter.ReadWord();

    Result result;
    PTXTypedOp(type,
        result = SetParamInternal<_PtxType_>(pExecutor, valueFullName, ptrFullName);
    )

    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Setting value to pointer " + ptrFullName + " failed" };
    }
    return {};
}

template<bool copyAsReference>
Result DispatchTable::CopyVarInternal(const ThreadExecutor* pExecutor,
                                      InstructionRunner::InstructionIter& iter) {

    const auto typeStr = iter.ReadWord();
    const auto type    = Types::StrToPTXType(typeStr);

    const auto dstFullName = iter.ReadWord();
    const auto srcFullName = iter.ReadWord();
    const auto dstDesc = Parser::ParseVectorName(dstFullName);
    const auto srcDesc = Parser::ParseVectorName(srcFullName);

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->GetTable()->FindVar(dstDesc.name)) {
        return { "Variable \"" + dstFullName + "\" storing the destination copy value is undefined" };
    }
    if (!pExecutor->GetTable()->FindVar(srcDesc.name)) {
        return { "Variable \"" + srcFullName + "\" storing the source copy value is undefined" };
    }
#endif

    decltype(auto) srcVar = pExecutor->GetTable()->GetVar(srcDesc.name);
    decltype(auto) dstVar = pExecutor->GetTable()->GetVar(dstDesc.name);

    bool result;
    PTXTypedOp(type,
        result = dstVar.AssignValue<_PtxType_, copyAsReference>(srcVar, srcDesc.key, dstDesc.key);
    )

    if (!result) {
        return { "Failed to perform copy operation for virtual variables \"" + dstFullName + "\" and "
                 "\"" + srcFullName + "\"" };
    }
    return {};
}

Result DispatchTable::CopyVarAsReference(const ThreadExecutor* pExecutor,
                                         InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // cvta.to.global.u64   %rd5,   %rd3;

    auto result = CopyVarInternal<true>(pExecutor, iter);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Copying of variable failed" };
    }
    return {};
}

Result DispatchTable::CopyVarAsValue(const ThreadExecutor* pExecutor,
                                     InstructionRunner::InstructionIter& iter) {

    // Sample instruction:
    // mov.u32   %rd5,   %rd3;

    auto result = CopyVarInternal<false>(pExecutor, iter);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Copying of variable failed" };
    }
    return {};
}
