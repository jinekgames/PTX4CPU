#include "runner.h"

#include <parser.h>

#include <sstream>


namespace PTX4CPU {
namespace DispatchTable {

using pfnOpProc = Result(*)(Types::PTXType,
                            const std::vector<Types::ArgumentPair>&);

Result Op2Internal(ThreadExecutor* pExecutor,
                   const Types::Instruction& instruction,
                   pfnOpProc OpProc) {

    const auto type = instruction.GetPtxType();
    const auto args = pExecutor->RetrieveArgs(type, instruction.args);

#ifdef COMPILE_SAFE_CHECKS
    if (args.size() < 1 || !args[0].first) {
        return { "Variable storing the operation product is undefined" };
    }
    if (args.size() < 2 || !args[1].first) {
        return { "Variable storing the left operation argument is undefined" };
    }
    if (args.size() < 3 || !args[2].first) {
        return { "Variable storing the right operation argument is undefined" };
    }
    if (args.size() > 3) {
        PRINT_E("Too much arguments passed");
    }
#endif  // #ifdef COMPILE_SAFE_CHECKS

    return OpProc(type, args);
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
Result MulOp(Types::PTXType type,
             const std::vector<Types::ArgumentPair>& args) {

    const auto& pDst     = args[0].first;
    const auto& pLeft    = args[1].first;
    const auto& pRght    = args[2].first;

    const auto  dstKey  = args[0].second;
    const auto  leftKey = args[1].second;
    const auto  rghtKey = args[2].second;

    PTXTypedOp(type,
        return MulOpTyped<mode, _PtxType_>(*pDst, *pLeft, *pRght,
                                           dstKey, leftKey, rghtKey);
    )
    return { "Invalid multiplication type" };
}

Result MulHi(ThreadExecutor* pExecutor, const Types::Instruction &instruction) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, instruction, MulOp<MulMode::Hi>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result MulLo(ThreadExecutor* pExecutor, const Types::Instruction &instruction) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, instruction, MulOp<MulMode::Lo>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result MulWide(ThreadExecutor* pExecutor, const Types::Instruction &instruction) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, instruction, MulOp<MulMode::Wide>);
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

Result AddOp(Types::PTXType type,
             const std::vector<Types::ArgumentPair>& args) {

    const auto& pDst     = args[0].first;
    const auto& pLeft    = args[1].first;
    const auto& pRght    = args[2].first;

    const auto  dstKey  = args[0].second;
    const auto  leftKey = args[1].second;
    const auto  rghtKey = args[2].second;

    PTXTypedOp(type,
        return AddOpTyped<_PtxType_>(*pDst, *pLeft, *pRght,
                                     dstKey, leftKey, rghtKey);
    )
    return { "Invalid multiplication type" };
}

Result Add(ThreadExecutor* pExecutor, const Types::Instruction& instruction) {

    // Instruction sample:
    //  add.s32   %rd7,   %r1,   4

    auto result = Op2Internal(pExecutor, instruction, AddOp);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}

}  // namespace DispatchTable
}  // namespace PTX4CPU
