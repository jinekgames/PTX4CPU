#include "runner.h"

#include <parser.h>

#include <type_traits>
#include <sstream>


namespace PTX4CPU {
namespace DispatchTable {

using pfnOpProc = Result(*)(Types::PTXType,
                            std::vector<Types::ArgumentPair>&);

Result Op2Internal(ThreadExecutor* pExecutor,
                   const Types::Instruction& instruction,
                   pfnOpProc OpProc) {

    const auto type = instruction.GetPtxType();
    auto args = pExecutor->RetrieveArgs(type, instruction.args);

#ifdef OPT_COMPILE_SAFE_CHECKS
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
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    return OpProc(type, args);
}

enum class MulMode {
    Hi,
    Lo,
    Wide
};

template<MulMode mode, Types::PTXType type>
Result MulOpTyped(Types::ArgumentPair& dst, Types::ArgumentPair& left,
                  Types::ArgumentPair& rght) {

    Result result;

    using MulResType   = Types::getVarType<Types::GetDoubleSizeType(type)>;
    using FinalResType = std::conditional_t<(mode == MulMode::Wide),
                                            MulResType,
                                            Types::getVarType<type>>;

    const auto mulResult =
        static_cast<MulResType>(Types::PTXVar::Get<type>(left)) *
        static_cast<MulResType>(Types::PTXVar::Get<type>(rght));

    const void* pValue = nullptr;

    if constexpr        (mode == MulMode::Wide) {
        pValue = &mulResult;
    } else if constexpr (mode == MulMode::Lo) {
        pValue = reinterpret_cast<const FinalResType*>(&mulResult);
    } else if constexpr (mode == MulMode::Hi) {
        pValue = reinterpret_cast<const FinalResType*>(&mulResult) + 1;
    } else {
        static_assert("invalid mul type");
    }

    if (!Types::PTXVar::AssignValue(dst, pValue)) {
        result = { "Failed to assign multiplication product" };
    }

    return result;
}

template<MulMode mode>
Result MulOp(Types::PTXType type,
             std::vector<Types::ArgumentPair>& args) {

#ifdef OPT_COMPILE_SAFE_CHECKS
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
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    auto& dst  = args[0];
    auto& left = args[1];
    auto& rght = args[2];

    PTXTypedOp(type,
        return MulOpTyped<mode, _PtxType_>(dst, left, rght);
    )
}

Result MulHi(ThreadExecutor* pExecutor, const Types::Instruction &instruction) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    const auto result =
        Op2Internal(pExecutor, instruction, MulOp<MulMode::Hi>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result MulLo(ThreadExecutor* pExecutor, const Types::Instruction &instruction) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    const auto result =
        Op2Internal(pExecutor, instruction, MulOp<MulMode::Lo>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}
Result MulWide(ThreadExecutor* pExecutor, const Types::Instruction &instruction) {

    // Instruction sample:
    //  mul.hi.s32   %rd7,   %r1,   4

    const auto result =
        Op2Internal(pExecutor, instruction, MulOp<MulMode::Wide>);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}

template<Types::PTXType type>
Result AddOpTyped(Types::ArgumentPair& dst, Types::ArgumentPair& left,
                  Types::ArgumentPair& rght) {

    using AddResType = Types::getVarType<type>;

    Types::PTXVar::Get<type>(dst) =
        static_cast<AddResType>(Types::PTXVar::Get<type>(left)) +
        static_cast<AddResType>(Types::PTXVar::Get<type>(rght));

    return {};
}

Result AddOp(Types::PTXType type,
             std::vector<Types::ArgumentPair>& args) {

#ifdef OPT_COMPILE_SAFE_CHECKS
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
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    auto& dst  = args[0];
    auto& left = args[1];
    auto& rght = args[2];

    PTXTypedOp(type,
        return AddOpTyped<_PtxType_>(dst, left, rght);
    )
    return { "Invalid multiplication type" };
}

Result Add(ThreadExecutor* pExecutor, const Types::Instruction& instruction) {

    // Instruction sample:
    //  add.s32   %rd7,   %r1,   4

    const auto result = Op2Internal(pExecutor, instruction, AddOp);
    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}

enum class CompareType : uint32_t {
    GreaterOrEqual,
};

template<CompareType compType, class RealType,
         std::enable_if_t<std::is_arithmetic_v<RealType>, int> = 0>
inline bool CompareReal(RealType left, RealType rght) {

    if constexpr (compType == CompareType::GreaterOrEqual) {
        return static_cast<bool>(left >= rght);
    } else {
        static_assert(false, "Invalid compare type");
    }
}

template<CompareType compType, Types::PTXType type>
Result CompareWrite(Types::ArgumentPair& pred, Types::ArgumentPair& left,
                    Types::ArgumentPair& rght) {

    constexpr auto ResultType = Types::PTXType::Pred;
    using ArgRealType = Types::getVarType<type>;

#ifdef OPT_COMPILE_SAFE_CHECKS
    if (pred.first->GetPTXType() != ResultType) {
        return { "Variable storing the compare product is not predicate" };
    }
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    Types::PTXVar::Get<ResultType>(pred) = CompareReal<compType, ArgRealType>(
        Types::PTXVar::Get<type>(left), Types::PTXVar::Get<type>(rght));

    return {};
}

Result LogicalGE(ThreadExecutor* pExecutor,
                 const Types::Instruction& instruction) {

    // Instruction sample:
    //  setp.ge.s32   %p1,   %r1,   %r2

    const auto type = instruction.GetPtxType();
    auto args = pExecutor->RetrieveArgs(type, instruction.args);

#ifdef OPT_COMPILE_SAFE_CHECKS
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
#endif  // #ifdef OPT_COMPILE_SAFE_CHECKS

    auto& pred = args[0];
    auto& left = args[1];
    auto& rght = args[2];

    Result result;

    PTXTypedOp(type,
        result = CompareWrite<CompareType::GreaterOrEqual, _PtxType_>(
            pred, left, rght);
    )

    if (!result) {
        PRINT_E("%s", result.msg.c_str());
        return { "Multiplication operation failed" };
    }
    return {};
}


}  // namespace DispatchTable
}  // namespace PTX4CPU
