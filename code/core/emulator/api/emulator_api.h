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


#include <translator_interface.h>

#include <string>


/**
 * Create translator object
 *
 * @param translator  pointer to poiner where object will be put
 * @param source      source code of a PTX
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateTranslator(PTX4CPU::ITranslator** translator, const std::string& source);

struct PtxInputData;
using PtxExecArgs = PtxInputData*;

/**
 * Create translator object
 *
 * @param inputData object where PTX execution arguments and temporary
 * variables will be put
 * @param jsonStr   content of a .json with execution arguments
*/
extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_ParseArgsJson(PtxExecArgs* inputData, const std::string& jsonStr);
