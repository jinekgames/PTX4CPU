#include "runner.h"

#include <parser.h>
#include <string_utils.h>

#include <charconv>

#ifdef COMPILE_SAFE_CHECKS
#ifdef WIN32
#include <Windows.h>
#endif  // #ifdef WIN32
#endif  // #ifdef COMPILE_SAFE_CHECKS


namespace PTX4CPU {
namespace DispatchTable {

template<Types::PTXType ptxType>
uint64_t RegisterMemoryInternal(ThreadExecutor* pExecutor,
                                const std::string& name,
                                const uint64_t count) {

    size_t i;
    for (i = 1; i <= count; ++i) {
        const std::string numberedName = name + std::to_string(i);
        pExecutor->GetTable()->AppendVar<ptxType>(numberedName);
        PRINT_V("Reg var %s : %s",
                name.c_str(),
                std::to_string(
                    pExecutor->GetTable()->GetVar(numberedName)).c_str());
    }
    return i - 1;
}

Result RegisterMemory(ThreadExecutor* pExecutor,
                      const Types::Instruction& instruction) {

    // Sample instruction:
    // .reg   .b32   %r<5>;

#ifdef COMPILE_SAFE_CHECKS
    if (instruction.args.size() < 1) {
        PRINT_E("Missed variable registration type");
    }
    if (instruction.args.size() < 2) {
        PRINT_E("Missed variable registration name");
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    const auto& typeStr = instruction.args[0];
    const auto  type    = Types::StrToPTXType(typeStr);

    const auto& nameWithCount = instruction.args[1];

    // Parse real name and count
    StringIteration::SmartIterator nameIter{nameWithCount};

    const auto name = nameIter.ReadWord();

    const auto countStr = nameIter.ReadWord();
    uint64_t count = 0;
    std::from_chars(countStr.data(), countStr.data() + countStr.length(),
                    count);

    uint64_t allocatedCount = 0;
    PTXTypedOp(type,
        allocatedCount =
            RegisterMemoryInternal<_PtxType_>(pExecutor, name, count);
    )

    if (allocatedCount != count)
        return { "Failed to allocate " +
                 std::to_string(count - allocatedCount) + " variables" };
    return {};
}

namespace {

template<Types::PTXType ptxType>
Result Dereference(const Types::PTXVar &ptrVar,
                   Types::getVarType<ptxType>& value,
                   char ptrKey = 'x') {

    using realType = std::remove_cvref_t<decltype(value)>;

    constexpr auto ptrPtxType = Types::GetSystemPtrType();

    auto ptr = ptrVar.Get<ptrPtxType>(ptrKey);

#ifdef COMPILE_SAFE_CHECKS
    if (!ptr) {
        return { "Null pointer dereference" };
    }
#ifdef WIN32
    if (IsBadReadPtr(reinterpret_cast<void*>(ptr),
                     static_cast<UINT_PTR>(sizeof(realType)))) {
        return { "Invalid pointer. Possible reading access violation" };
    }
#endif  // #ifdef WIN32
#endif  // #ifdef COMPILE_SAFE_CHECKS

    value = *reinterpret_cast<realType*>(ptr);

    return {};
}

template<Types::PTXType ptxType>
Result DereferenceAndSet(Types::PTXVar* ptrVar,
                         const Types::getVarType<ptxType>& value,
                         char ptrKey = 'x') {

    using realType = std::remove_cvref_t<decltype(value)>;

    constexpr auto PtrType = Types::GetSystemPtrType();

    auto ptr = reinterpret_cast<realType*>(ptrVar->Get<PtrType>(ptrKey));

#ifdef COMPILE_SAFE_CHECKS
    if (!ptr) {
        return { "Null pointer dereference" };
    }
#ifdef WIN32
    if (IsBadReadPtr(reinterpret_cast<void*>(ptr),
                     static_cast<UINT_PTR>(sizeof(realType)))) {
        return { "Invalid pointer. Possible reading access violation" };
    }
#endif  // #ifdef WIN32
#endif  // #ifdef COMPILE_SAFE_CHECKS

    *ptr = value;

    return {};
}

std::string RemoveVarNameBrackets(const std::string& name)
{
    constexpr auto delimiter = StringIteration::AllSpaces |
                               StringIteration::Brackets;
    return StringIteration::SmartIterator{name}.ReadWord(false, delimiter);
}

}  // anonimous namespace

template<Types::PTXType ptxType>
Result LoadParamInternal(ThreadExecutor* pExecutor,
                         const std::string& valueFullName,
                         const std::string& ptrFullName) {

    const auto valueDesc = Parser::ParseVectorName(valueFullName);
    const auto ptrDesc   = Parser::ParseVectorName(ptrFullName);

    auto pPtrVar = pExecutor->GetTable()->FindVar(ptrDesc.name);

#ifdef COMPILE_SAFE_CHECKS
    if (!pPtrVar) {
        return { "Variable \"" + ptrFullName + "\" storing the pointer is undefined" };
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    Types::getVarType<ptxType> value;
    auto result = Dereference<ptxType>(*pPtrVar, value, ptrDesc.key);

    if (!result) {
        PRINT_E(result.msg.c_str());
        return { "Loading of param " + ptrFullName + " failed" };
    }

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->GetTable()->FindVar(valueDesc.name)) {
        return { "Variable \"" + valueFullName + "\" storing the dereferanced value is undefined" };
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    pExecutor->GetTable()->GetValue<ptxType>(valueDesc.name, valueDesc.key) =
        value;

    return {};
}

Result LoadParam(ThreadExecutor* pExecutor,
                 const Types::Instruction& instruction) {

    // Sample instruction:
    // ld.param.u64   %rd1,   [_Z9addKernelPiPKiS1__param_0];

    const auto type = instruction.GetPtxType();

#ifdef COMPILE_SAFE_CHECKS
    if (instruction.args.size() < 1) {
        PRINT_E("Missed destination `ld` argumemt");
    }
    if (instruction.args.size() < 2) {
        PRINT_E("Missed source `ld` argumemt");
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    const auto  valueFullName = instruction.args[0];
    const auto& ptrFullName   = RemoveVarNameBrackets(instruction.args[1]);

    Result result;
    PTXTypedOp(type,
        result =
            LoadParamInternal<_PtxType_>(pExecutor, valueFullName, ptrFullName);
    )

    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Dereference of " + ptrFullName + " failed" };
    }
    return {};
}

template<Types::PTXType ptxType>
Result SetParamInternal(ThreadExecutor* pExecutor,
                        const std::string& valueFullName,
                        const std::string& ptrFullName) {

    const auto valueDesc = Parser::ParseVectorName(valueFullName);
    const auto ptrDesc   = Parser::ParseVectorName(ptrFullName);

    auto ptrVar = pExecutor->GetTable()->FindVar(ptrDesc.name);
#ifdef COMPILE_SAFE_CHECKS
    if (!ptrVar) {
        return { "Variable \"" + ptrFullName + "\" storing the pointer is undefined" };
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    auto value = pExecutor->GetTable()->GetValue<ptxType>(valueDesc.name, valueDesc.key);
    auto result = DereferenceAndSet<ptxType>(ptrVar.get(), value, ptrDesc.key);

    if (!result) {
        PRINT_E(result.msg.c_str());
        return { "Loading of param " + ptrFullName + " failed" };
    }

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->GetTable()->FindVar(valueDesc.name)) {
        return { "Variable \"" + valueFullName + "\" storing the dereferanced value is undefined" };
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    return {};
}

Result SetParam(ThreadExecutor* pExecutor,
                const Types::Instruction& instruction) {

    // Sample instruction:
    // st.global.u32   [%rd10],   %r4;

    const auto type = instruction.GetPtxType();

#ifdef COMPILE_SAFE_CHECKS
    if (instruction.args.size() < 1) {
        PRINT_E("Missed destination `ld` argumemt");
    }
    if (instruction.args.size() < 2) {
        PRINT_E("Missed source `ld` argumemt");
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    const auto& ptrFullName   = RemoveVarNameBrackets(instruction.args[0]);
    const auto  valueFullName = instruction.args[1];

    Result result;
    PTXTypedOp(type,
        result = SetParamInternal<_PtxType_>(
                     pExecutor, valueFullName, ptrFullName);
    )

    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Setting value to pointer " + ptrFullName + " failed" };
    }
    return {};
}

template<bool copyAsReference>
Result CopyVarInternal(ThreadExecutor* pExecutor,
                       const Types::Instruction& instruction) {

    const auto type = instruction.GetPtxType();
    auto args = pExecutor->RetrieveArgs(type, instruction.args);

#ifdef COMPILE_SAFE_CHECKS
    if (args.size() < 1 || !args[0].first) {
        return { "Variable \"" + instruction.args[0] + "\" storing the destination copy value is undefined" };
    }
    if (args.size() < 2 || !args[1].first) {
        return { "Variable \"" + instruction.args[1] + "\" storing the source copy value is undefined" };
    }
    if (args.size() > 2) {
        PRINT_E("Too much arguments passed");
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    auto& dst = args[0];
    auto& src = args[1];

    bool result;
    PTXTypedOp(type,
        result =
            Types::PTXVar::AssignValue<_PtxType_, copyAsReference>(dst, src);
    )

    if (!result) {
        return { "Failed to perform copy operation for virtual variables" };
    }
    return {};
}

Result CopyVarAsReference(ThreadExecutor* pExecutor,
                          const Types::Instruction& instruction) {

    // Sample instruction:
    // cvta.to.global.u64   %rd5,   %rd3;

    auto result = CopyVarInternal<true>(pExecutor, instruction);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Copying of variable failed" };
    }
    return {};
}

Result CopyVarAsValue(ThreadExecutor* pExecutor,
                      const Types::Instruction& instruction) {

    // Sample instruction:
    // mov.u32   %rd5,   %rd3;

    auto result = CopyVarInternal<false>(pExecutor, instruction);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Copying of variable failed" };
    }
    return {};
}

}  // namespace DispatchTable
}  // namespace PTX4CPU
