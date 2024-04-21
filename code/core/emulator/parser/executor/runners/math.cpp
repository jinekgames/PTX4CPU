#include "runner.h"

#include <parser.h>

#include <sstream>


using namespace PTX4CPU;

template<Types::PTXType type>
Types::PTXVarPtr DispatchTable::CreateTempValueVarTyped(const std::string& value) {

    std::stringstream ss(value);
    Types::getVarType<type> realValue;
    ss >> realValue;

    return Types::PTXVarPtr{new Types::PTXVarTyped<type>{&realValue}};
}

Types::PTXVarPtr DispatchTable::CreateTempValueVar(Types::PTXType type, const std::string& value) {

    PTXTypedOp(type,
        return  CreateTempValueVarTyped<_Runtime_Type_>(value);
    )
    return nullptr;
}

Result DispatchTable::Op2Internal(const ThreadExecutor* pExecutor,
                                  InstructionRunner::InstructionIter& iter,
                                  pOpProc OpProc) {

    const auto typeStr = iter.ReadWord2();
    const auto type = Types::GetFromStr(typeStr);

    const auto dstFullName  = iter.ReadWord2();
    const auto leftFullName = iter.ReadWord2();
    const auto rghtFullName = iter.ReadWord2();

    const auto dstDesc  = Parser::ParseVectorName(dstFullName);
    const auto leftDesc = Parser::ParseVectorName(leftFullName);
    const auto rghtDesc = Parser::ParseVectorName(rghtFullName);

    auto pDstVar  = pExecutor->GetTable()->FindVar(dstDesc.name);
    auto pLeftVar = pExecutor->GetTable()->FindVar(leftDesc.name);
    auto pRghtVar = pExecutor->GetTable()->FindVar(rghtDesc.name);

    // Make a vars for not existed vars
    Types::PTXVarPtr pLeftTempVar;
    Types::PTXVarPtr pRghtTempVar;
    const auto varNamePrefix = '%';
    if (!pLeftVar && leftFullName.front() != varNamePrefix) {
        pLeftTempVar = std::move(CreateTempValueVar(type, leftFullName));
        pLeftVar = pLeftTempVar.get();
    }
    if (!pRghtVar && rghtFullName.front() != varNamePrefix) {
        pRghtTempVar = std::move(CreateTempValueVar(type, rghtFullName));
        pRghtVar = pRghtTempVar.get();
    }

#ifdef COMPILE_SAFE_CHECKS
    if (!pDstVar) {
        return { "Variable \"" + dstFullName + "\" storing the operation product is undefined" };
    }
    if (!pLeftVar) {
        return { "Variable \"" + leftFullName + "\" storing the operation argument is undefined" };
    }
    if (!pRghtVar) {
        return { "Variable \"" + rghtFullName + "\" storing the operation argument is undefined" };
    }
#endif

    return OpProc(type, *pDstVar, *pLeftVar, *pRghtVar,
                  dstDesc.key, leftDesc.key, rghtDesc.key);
}

enum class MulMode {
    Hi,
    Lo,
    Wide
};

template<MulMode mode, Types::PTXType type>
Result MulOpTyped(Types::PTXVar& dst, Types::PTXVar& left, Types::PTXVar& rght,
                  char dstKey = 'x', char leftKey = 'x', char rghtKey = 'x') {

    Result result;

    using MulResType   = Types::getVarType<Types::GetDoubleSizeType(type)>;
    using FinalResType = std::conditional_t<mode == MulMode::Wide, MulResType, Types::getVarType<type>>;

    decltype(auto) mulResult = static_cast<MulResType>(left.Get<type>(leftKey)) *
                               static_cast<MulResType>(rght.Get<type>(rghtKey));

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

    return result;
}

template<MulMode mode>
Result MulOp(Types::PTXType type, Types::PTXVar& dst, Types::PTXVar& left, Types::PTXVar& rght,
             char dstKey = 'x', char leftKey = 'x', char rghtKey = 'x') {

    PTXTypedOp(type,
        return MulOpTyped<mode, _Runtime_Type_>(dst, left, rght, dstKey, leftKey, rghtKey);
    )
    return { "Invalid multiplication type" };
}

Result DispatchTable::MulHi(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, iter, MulOp<MulMode::Hi>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result DispatchTable::MulLo(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, iter, MulOp<MulMode::Lo>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result DispatchTable::MulWide(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, iter, MulOp<MulMode::Wide>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}

template<Types::PTXType type>
Result AddOpTyped(Types::PTXVar& dst, Types::PTXVar& left, Types::PTXVar& rght,
                  char dstKey = 'x', char leftKey = 'x', char rghtKey = 'x') {

    using AddResType = Types::getVarType<type>;

    dst.Get<type>(dstKey) = static_cast<AddResType>(left.Get<type>(leftKey)) +
                            static_cast<AddResType>(rght.Get<type>(rghtKey));

    return {};
}

Result AddOp(Types::PTXType type, Types::PTXVar& dst, Types::PTXVar& left, Types::PTXVar& rght,
             char dstKey = 'x', char leftKey = 'x', char rghtKey = 'x') {

    PTXTypedOp(type,
        return AddOpTyped<_Runtime_Type_>(dst, left, rght, dstKey, leftKey, rghtKey);
    )
    return { "Invalid multiplication type" };
}

Result DispatchTable::Add(const ThreadExecutor* pExecutor, InstructionRunner::InstructionIter& iter) {

    // Instruction sample:
    //  add.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, iter, AddOp);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
