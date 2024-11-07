#pragma once

#include <utils/api_types.h>
#include <utils/result.h>

#include "json_parser/parser.h"


namespace PTX4CPU {

/**
 * Parces a given runtime CUDA args according to the given kernal arguments
 * destricption
 *
 * @param ppArgs     pointer to an array of runtime CUDA arguments
 * @param kernelArgs kernel arguments description
 * @param pInputData pointer to object where PTX execution arguments and
 * temporary variables will be put
 *
 * @return Parsing result
*/
Result ParseCudaArgs(const void* const* ppArgs,
                     Types::Function::Arguments& kernelArgs,
                     Types::PtxInputData* pInputData);

}  // namespace PTX4CPU
