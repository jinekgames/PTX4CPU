#include "arg_parsers.h"

#include <logger/logger.h>
#include <parser_types.h>
#include <utils/string_utils.h>
#include <utils/validator.h>

#include <type_traits>


namespace PTX4CPU {

namespace {

template<Types::PTXType type>
void InsertTypedArg(const void* const ptxArg, Types::PtxInputData& inputData) {

    constexpr auto ptxPtrType = Types::GetSystemPtrType();

    using realType = Types::getVarType<ptxPtrType>;
    // variable storing the address, which is a ptx kernel argument
    const auto* pPtxArgReal = reinterpret_cast<const realType*>(&ptxArg);

    Validator::CheckPointer(pPtxArgReal);

    // Pass to the virtual var constructor pointer to the value,
    // which need to be stored in the variables
    Types::PTXVar* pPtxArg = new Types::PTXVarTyped<ptxPtrType>(pPtxArgReal);

    inputData.execArgs.emplace_back(pPtxArg);
}

}  // anonimous namespace

Result ParseCudaArgs(const void* const* ppArgs,
                     Types::Function::Arguments& kernelArgs,
                     PtxExecArgs* ppInputData) {

    const auto argsSize = kernelArgs.size();

    if (!ppArgs) {
        if(argsSize != 0) {
            return { FormatString("Null Arguments array passed while kernel "
                                  "require {} arguments", argsSize) };
        }
        return { "Null Arguments array passed" };
    }

    auto& pInputData = *ppInputData;
    pInputData       = new Types::PtxInputData{};
    auto& inputData  = *pInputData;

    size_t i = 0;
    for (const auto& it : kernelArgs) {
        const auto& argDesc   = it.second;
        // ptr to c++ kernel arg. real ptx kernel arg
        auto*       pArgValue = ppArgs[i];

        const auto type = argDesc.type;

        PTXTypedOp(type,
            InsertTypedArg<_PtxType_>(pArgValue, inputData);
        );

        ++i;
    }

    return {};
}

}  // namespace PTX4CPU
