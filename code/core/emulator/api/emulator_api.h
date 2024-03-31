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
