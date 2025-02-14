#pragma once

#if defined(WIN32)
#define EMULATOR_EXPORT_API __declspec(dllexport)
#define EMULATOR_IMPORT_API __declspec(dllimport)
#define EMULATOR_CC __cdecl
#else
#define EMULATOR_EXPORT_API __attribute__((visibility("default"), used))
#define EMULATOR_IMPORT_API
#define EMULATOR_CC
#endif

#ifdef EMULATORLIB_EXPORTS
#define EMULATOR_API EMULATOR_EXPORT_API
#else
#define EMULATOR_API EMULATOR_IMPORT_API
#endif


#include "emulator/emulator_interface.h"

#include <string>


/** @brief Creates emulator object
 *
 * @param ppEmulator a double poiner where object will be put
 * @param sourceCode source code of a PTX
 *
 * @note `*ppEmulator` should be destoryed with `EMULATOR_DestroyEmulator()`
 * @note Puts `nullptr` in case of failure
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateEmulator(
    PTX4CPU::IEmulator** ppEmulator,
    const std::string&   sourceCode);

/** @brief Destroys emulator object
 *
 * @param pEmulator object to be destroyed
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_DestroyEmulator(
    PTX4CPU::IEmulator* pEmulator);

/** @brief Processes PTX arguments from CUDA runtime
 *
 * @param pInputData a pointer to PtxExecArgs where the processing result will
 * be put
 * @param kernel     descriptor of the kernel to pass arguments to.
 * Could be retrived from Emulator object using
 * `IEmulator::GetKernelDescriptor()` API
 * @param ppArgs     arguments passed to CUDA runtime as an array of `void*`
 * pointers
 *
 * @note `*pInputData` should be destoryed with `EMULATOR_DestroyArgs()`
 * @note Puts `PTX4CPU_NULL_HANDLE` in case of failure
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateArgs(
    PTX4CPU::PtxExecArgs*            pInputData,
    const PTX4CPU::PtxFuncDescriptor kernel,
    const void* const*               ppArgs);

/** @brief Parses PTX arguments from json
 *
 * @param pInputData object where PTX execution arguments and temporary
 * variables will be put
 * @param jsonStr    content of a .json with execution arguments
 *
 * @note `*pInputData` should be destoryed with `EMULATOR_DestroyArgs()`
 * @note Puts `PTX4CPU_NULL_HANDLE` in case of failure
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateArgsJson(
    PTX4CPU::PtxExecArgs* pInputData,
    const std::string&    jsonStr);

/** @brief Destroys PTX arguments
 *
 * @param inputData object to be destoryed
 */
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_DestroyArgs(
    PTX4CPU::PtxExecArgs inputData);

/** @brief Serializes PTX arguments into json
 *
 * @param inputData object where PTX execution result and temporary
 * variables will be put
 * @param jsonStr   an output .json with execution resuts
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_SerializeArgsJson(
    PTX4CPU::PtxExecArgs inputData,
    std::string&         jsonStr);
