#include "ext_parsers.h"

#include <logger/logger.h>
#include <parser_types.h>
#include <utils/string_utils.h>


template<Types::PTXType type>
void InsertTypedArg(const void* const pArg, PtxInputData& inputData);


Result ParseCudaArgs(const void* const* ppArgs,
                     Types::Function::Arguments& kernelArgs,
                     PtxInputData& inputData) {

    const auto argsSize = kernelArgs.size();

    if (!ppArgs) {
        if(argsSize != 0) {
            return { FormatString("Null Arguments array passed while kernel "
                                  "require {} arguments", argsSize) };
        }
        return { "Null Arguments array passed" };
    }

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