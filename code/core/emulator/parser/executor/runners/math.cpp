#include "runner.h"

#include <parser.h>


using namespace PTX4CPU;

Result DispatchTable::Op2Internal(const ThreadExecutor* pExecutor,
                                  InstructionRunner::InstructionIter& iter,
                                  pOpProc OpProc) {

    // Instruction sample:
    //  mul.wide.s32   %rd7,   %r1,   4

    const auto typeStr = iter.ReadWord2();
    const auto type = Types::GetFromStr(typeStr);

    const auto dstFullName  = iter.ReadWord2();
    const auto leftFullName = iter.ReadWord2();
    const auto rghtFullName = iter.ReadWord2();

#ifdef COMPILE_SAFE_CHECKS
    bool leftGenned = false;
    bool rghtGenned = false;
#endif

    // Make a vars for not existed vars
    if (leftFullName.front() != '%')

    const auto dstDesc  = Parser::ParseVectorName(dstFullName);
    const auto leftDesc = Parser::ParseVectorName(leftFullName);
    const auto rghtDesc = Parser::ParseVectorName(rghtFullName);

#ifdef COMPILE_SAFE_CHECKS
    if (!pExecutor->GetTable()->FindVar(dstDesc.name)) {
        return { "Variable \"" + dstFullName + "\" storing the operation product is undefined" };
    }
    if (!leftGenned && !pExecutor->GetTable()->FindVar(leftDesc.name)) {
        return { "Variable \"" + leftFullName + "\" storing the operation argument is undefined" };
    }
    if (!rghtGenned && !pExecutor->GetTable()->FindVar(rghtDesc.name)) {
        return { "Variable \"" + rghtFullName + "\" storing the operation argument is undefined" };
    }
#endif

    decltype(auto) dstVar  = pExecutor->GetTable()->GetVar(dstDesc.name);
    decltype(auto) leftVar = pExecutor->GetTable()->GetVar(leftDesc.name);
    decltype(auto) rghtVar = pExecutor->GetTable()->GetVar(rghtDesc.name);

    return OpProc(type, dstVar, leftVar, rghtVar,
                  dstDesc.key, leftDesc.key, rghtDesc.key);

    return { "Invalid operation runner" };
}

enum class MulMode {
    Hi,
    Lo,
    Wide
};

template<MulMode mode>
Result MulOp(Types::PTXType type, Types::PTXVar& dst, Types::PTXVar& left, Types::PTXVar& rght,
             char dstKey = 'x', char leftKey = 'x', char rghtKey = 'x') {

    Result result;

    PTXTypedOp(type,

        using MulResType   = Types::getVarType<Types::GetDoubleSizeType(_Runtime_Type_)>;
        using FinalResType = std::conditional_t<mode == MulMode::Wide, MulResType, Types::getVarType<_Runtime_Type_>>;

        decltype(auto) mulResult = static_cast<MulResType>(left.Get<_Runtime_Type_>(leftKey)) *
                                   static_cast<MulResType>(rght.Get<_Runtime_Type_>(rghtKey));

        if constexpr (mode == MulMode::Wide) {
            if (!dst.AssignValue(&mulResult, dstKey)) {
                result = { "Failed to assign multiplication product" };
            }
        } else if constexpr (mode == MulMode::Lo) {
            auto pRes = reinterpret_cast<FinalResType*>(&mulResult);
            auto res = *pRes;
            if (!dst.AssignValue(&res, dstKey)) {
                result = { "Failed to assign multiplication product" };
            }
        } else { // mode == MulMode::Hi
            auto pRes = reinterpret_cast<FinalResType*>(&mulResult) + 1;
            auto res = *pRes;
            if (!dst.AssignValue(&res, dstKey)) {
                result = { "Failed to assign multiplication product" };
            }
        }
    )

    return result;
}

Result DispatchTable::MulHi(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {
    auto result = Op2Internal(pExecutor, iter, MulOp<MulMode::Hi>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result DispatchTable::MulLo(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {
    auto result = Op2Internal(pExecutor, iter, MulOp<MulMode::Lo>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result DispatchTable::MulWide(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {
    auto result = Op2Internal(pExecutor, iter, MulOp<MulMode::Wide>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
