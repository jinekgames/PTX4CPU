#include "ext_parsers.h"

#include <logger/logger.h>
#include <parser_types.h>
#include <utils/string_utils.h>


namespace PTX4CPU {

namespace {

template<Types::PTXType type>
void InsertTypedArg(const void* const pArg, Types::PtxInputData& inputData) {

    using RealType = Types::getVarType<type>;
    decltype(auto) pArgReal = reinterpret_cast<const RealType*>(pArg);

    Types::PTXVar* pPtxArg = new Types::PTXVarTyped<type>(pArgReal);

    inputData.execArgs.emplace_back(pPtxArg);
}

}  // anonimous namespace

Result ParseCudaArgs(const void* const* ppArgs,
                     Types::Function::Arguments& kernelArgs,
                     Types::PtxInputData** ppInputData) {

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
        auto& argDesc   = it.second;
        auto  pArgValue = ppArgs[i];

        const auto type = argDesc.type;

        PTXTypedOp(type,
            InsertTypedArg<_PtxType_>(pArgValue, inputData);
        );

        ++i;
    }

    return {};
}

}  // namespace PTX4CPU
