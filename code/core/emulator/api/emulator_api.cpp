#include <emulator_api.h>

#include <translator.h>

extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateTranslator(PTX2ASM::ITranslator** translator, const std::string& path) {

    *translator = new PTX2ASM::Translator(path);
}

