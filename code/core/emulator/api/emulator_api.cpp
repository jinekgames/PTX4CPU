#include <emulator_api.h>

#include <translator.h>


extern "C" EMULATOR_EXPORT_API void EMULATOR_CC
EMULATOR_CreateTranslator(PTX4CPU::ITranslator** translator, const std::string& source) {

    *translator = new PTX4CPU::Translator(source);
}
